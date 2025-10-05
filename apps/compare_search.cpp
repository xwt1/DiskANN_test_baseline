// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>

#include <boost/program_options.hpp>

#include "ann_exception.h"
#include "compare_common.hpp"
#include "index_factory.h"
#include "program_options_utils.hpp"
#include "utils.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;
using namespace compare_experiment;

struct SearchConfig
{
    std::string data_type;
    std::string dist_fn;
    std::string query_file;
    std::string gt_file{ "null" };
    std::string index_root;
    std::vector<uint32_t> shard_counts;
    std::vector<uint32_t> search_lists;
    uint32_t recall_at = 0;
    uint32_t per_shard_k = 0;
    uint32_t search_threads = 0;
    std::string output_dir;
};

struct LoadedShard
{
    uint32_t ordinal = 0;
    size_t point_count = 0;
    std::vector<uint32_t> mapping;
    std::unique_ptr<diskann::AbstractIndex> index;
};

struct EvaluationSummary
{
    double recall = std::numeric_limits<double>::quiet_NaN();
    double avg_comparisons = 0.0;
    uint64_t total_comparisons = 0;
};

struct Candidate
{
    uint32_t id = std::numeric_limits<uint32_t>::max();
    float distance = std::numeric_limits<float>::quiet_NaN();
};

uint32_t resolve_thread_count(uint32_t requested)
{
    return requested == 0 ? static_cast<uint32_t>(omp_get_num_procs()) : requested;
}

std::vector<LoadedShard> load_shards(const BuildManifest &manifest, const fs::path &shard_dir,
                                     diskann::Metric metric, uint32_t load_threads, uint32_t search_l)
{
    std::vector<LoadedShard> shards;
    shards.reserve(manifest.shards.size());

    auto build_params = diskann::IndexWriteParametersBuilder(manifest.build_L, manifest.build_R)
                             .with_alpha(manifest.build_alpha)
                             .with_num_threads(manifest.build_threads)
                             .with_saturate_graph(false)
                             .build();

    for (const auto &entry : manifest.shards)
    {
        fs::path index_path = shard_dir / entry.index_file;
        fs::path map_path = shard_dir / entry.map_file;
        if (!fs::exists(index_path))
        {
            std::stringstream ss;
            ss << "Index file not found: " << index_path;
            throw diskann::ANNException(ss.str(), -1);
        }
        if (!fs::exists(map_path))
        {
            std::stringstream ss;
            ss << "Mapping file not found: " << map_path;
            throw diskann::ANNException(ss.str(), -1);
        }

        auto config = diskann::IndexConfigBuilder()
                           .with_metric(metric)
                           .with_dimension(manifest.dimension)
                           .with_max_points(entry.point_count)
                           .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                           .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                           .with_data_type(manifest.data_type)
                           .with_label_type(type_name<uint32_t>())
                           .with_tag_type(type_name<uint32_t>())
                           .with_index_write_params(build_params)
                           .is_dynamic_index(false)
                           .is_enable_tags(false)
                           .is_concurrent_consolidate(false)
                           .is_pq_dist_build(false)
                           .is_use_opq(false)
                           .build();

        diskann::IndexFactory factory(config);
        auto index = factory.create_instance();
        index->load(index_path.string().c_str(), load_threads, search_l);
        index->reset_search_compute_stats();

        shards.push_back(LoadedShard{entry.ordinal, entry.point_count, load_mapping(map_path), std::move(index)});
    }

    return shards;
}

EvaluationSummary summarize_results(const std::vector<uint32_t> &results, size_t query_count, uint32_t recall_at,
                                    const uint32_t *gt_ids, float *gt_dists, uint32_t gt_dim,
                                    bool has_truth, uint64_t total_comparisons)
{
    EvaluationSummary summary;
    summary.total_comparisons = total_comparisons;
    summary.avg_comparisons = query_count == 0 ? 0.0 : static_cast<double>(total_comparisons) / query_count;

    if (has_truth)
    {
        summary.recall = diskann::calculate_recall(static_cast<uint32_t>(query_count), const_cast<uint32_t *>(gt_ids),
                                                   gt_dists, gt_dim, const_cast<uint32_t *>(results.data()), recall_at,
                                                   recall_at);
    }
    return summary;
}

template <typename T>
EvaluationSummary evaluate_shards(const std::vector<LoadedShard> &shards, diskann::Metric metric,
                                  const T *queries, size_t query_count, size_t query_aligned_dim,
                                  uint32_t recall_at, uint32_t per_shard_k, uint32_t search_L,
                                  const uint32_t *gt_ids, float *gt_dists, uint32_t gt_dim, bool has_truth,
                                  uint32_t search_threads)
{
    if (shards.empty())
    {
        throw diskann::ANNException("No shard indexes loaded.", -1);
    }
    if (per_shard_k == 0)
    {
        throw diskann::ANNException("per_shard_k must be positive.", -1);
    }

    const auto preference = preference_for_metric(metric);

    for (const auto &shard : shards)
    {
        shard.index->reset_search_compute_stats();
    }

    std::vector<uint32_t> final_results(query_count * recall_at, std::numeric_limits<uint32_t>::max());
    uint64_t total_comparisons = 0;

    omp_set_num_threads(search_threads);
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : total_comparisons)
    for (int64_t qi = 0; qi < static_cast<int64_t>(query_count); ++qi)
    {
        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(per_shard_k) * shards.size());
        uint64_t query_comparisons = 0;

        std::vector<uint32_t> local_ids(per_shard_k);
        std::vector<float> local_dists(per_shard_k);

        const T *query_ptr = queries + static_cast<size_t>(qi) * query_aligned_dim;
        for (const auto &shard : shards)
        {
            auto result = shard.index->search(query_ptr, per_shard_k, search_L, local_ids.data(), local_dists.data());
            query_comparisons += result.second;

            for (uint32_t k = 0; k < per_shard_k; ++k)
            {
                uint32_t local_id = local_ids[k];
                if (local_id == std::numeric_limits<uint32_t>::max())
                {
                    continue;
                }
                if (local_id >= shard.mapping.size())
                {
                    continue;
                }
                candidates.push_back(Candidate{shard.mapping[local_id], local_dists[k]});
            }
        }

        size_t keep = std::min<size_t>(recall_at, candidates.size());
        if (keep > 0)
        {
            if (preference == DistancePreference::Minimize)
            {
                std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                                  [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });
            }
            else
            {
                std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                                  [](const Candidate &a, const Candidate &b) { return a.distance > b.distance; });
            }
            for (size_t r = 0; r < keep; ++r)
            {
                final_results[static_cast<size_t>(qi) * recall_at + r] = candidates[r].id;
            }
        }
        total_comparisons += query_comparisons;
    }

    return summarize_results(final_results, query_count, recall_at, gt_ids, gt_dists, gt_dim, has_truth,
                              total_comparisons);
}

SearchConfig parse_arguments(int argc, char **argv)
{
    SearchConfig config;

    po::options_description desc{
        program_options_utils::make_program_description("compare_search", "Evaluate sharded indexes")};

    try
    {
        po::options_description required("Required");
        required.add_options()
            ("data_type", po::value<std::string>(&config.data_type)->required(),
             program_options_utils::DATA_TYPE_DESCRIPTION)
            ("dist_fn", po::value<std::string>(&config.dist_fn)->required(),
             program_options_utils::DISTANCE_FUNCTION_DESCRIPTION)
            ("query_file", po::value<std::string>(&config.query_file)->required(),
             program_options_utils::QUERY_FILE_DESCRIPTION)
            ("recall_at", po::value<uint32_t>(&config.recall_at)->required(),
             program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION)
            ("search_list", po::value<std::vector<uint32_t>>(&config.search_lists)->multitoken()->required(),
             program_options_utils::SEARCH_LIST_DESCRIPTION)
            ("shard_counts", po::value<std::vector<uint32_t>>(&config.shard_counts)->multitoken()->required(),
             "One or more shard counts to evaluate")
            ("index_root", po::value<std::string>(&config.index_root)->required(),
             "Root directory containing S_<count> subdirectories");

        po::options_description optional("Optional");
        optional.add_options()
            ("gt_file", po::value<std::string>(&config.gt_file)->default_value(std::string("null")),
             program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION)
            ("per_shard_k", po::value<uint32_t>(&config.per_shard_k)->default_value(0),
             "Number of candidates to retrieve from each shard (0 = recall_at)")
            ("search_threads", po::value<uint32_t>(&config.search_threads)->default_value(0),
             "Number of threads to use for querying (0 = auto)")
            ("output_dir", po::value<std::string>(&config.output_dir)->default_value(std::string("")),
             "Directory where per-shard CSV summaries will be written");

        desc.add_options()("help,h", "Print arguments");
        desc.add(required).add(optional);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            std::exit(0);
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        std::cerr << desc << std::endl;
        std::exit(-1);
    }

    return config;
}

template <typename T> int run_search(const SearchConfig &config)
{
    diskann::Metric metric = parse_metric(config.dist_fn, config.data_type);

    T *queries = nullptr;
    size_t query_count = 0;
    size_t query_dim = 0;
    size_t query_aligned_dim = 0;
    diskann::load_aligned_bin<T>(config.query_file, queries, query_count, query_dim, query_aligned_dim);

    if (query_count == 0)
    {
        diskann::aligned_free(queries);
        throw diskann::ANNException("Query file is empty.", -1);
    }

    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t gt_num = 0;
    size_t gt_dim = 0;
    const bool has_truth = !is_null_token(config.gt_file) && fs::exists(config.gt_file);
    if (has_truth)
    {
        diskann::load_truthset(config.gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_count)
        {
            diskann::aligned_free(queries);
            delete[] gt_ids;
            delete[] gt_dists;
            std::stringstream ss;
            ss << "Mismatch between query count (" << query_count << ") and ground-truth count (" << gt_num << ").";
            throw diskann::ANNException(ss.str(), -1);
        }
        if (config.recall_at > gt_dim)
        {
            diskann::aligned_free(queries);
            delete[] gt_ids;
            delete[] gt_dists;
            std::stringstream ss;
            ss << "recall_at (" << config.recall_at << ") exceeds ground-truth dimension (" << gt_dim << ").";
            throw diskann::ANNException(ss.str(), -1);
        }
    }

    if (config.shard_counts.empty())
    {
        diskann::aligned_free(queries);
        if (gt_ids != nullptr)
        {
            delete[] gt_ids;
        }
        if (gt_dists != nullptr)
        {
            delete[] gt_dists;
        }
        throw diskann::ANNException("At least one shard count must be provided.", -1);
    }

    std::vector<uint32_t> search_lists = config.search_lists;
    std::sort(search_lists.begin(), search_lists.end());
    search_lists.erase(std::unique(search_lists.begin(), search_lists.end()), search_lists.end());
    if (search_lists.empty())
    {
        diskann::aligned_free(queries);
        if (gt_ids != nullptr)
        {
            delete[] gt_ids;
        }
        if (gt_dists != nullptr)
        {
            delete[] gt_dists;
        }
        throw diskann::ANNException("At least one search_list value must be specified.", -1);
    }

    const uint32_t min_L = *std::min_element(search_lists.begin(), search_lists.end());
    if (config.recall_at > min_L)
    {
        diskann::aligned_free(queries);
        if (gt_ids != nullptr)
        {
            delete[] gt_ids;
        }
        if (gt_dists != nullptr)
        {
            delete[] gt_dists;
        }
        std::stringstream ss;
        ss << "recall_at (" << config.recall_at << ") cannot exceed the smallest search_list value (" << min_L
           << ").";
        throw diskann::ANNException(ss.str(), -1);
    }

    const uint32_t per_shard_k = config.per_shard_k == 0 ? config.recall_at : config.per_shard_k;
    if (per_shard_k > min_L)
    {
        diskann::aligned_free(queries);
        if (gt_ids != nullptr)
        {
            delete[] gt_ids;
        }
        if (gt_dists != nullptr)
        {
            delete[] gt_dists;
        }
        std::stringstream ss;
        ss << "per_shard_k (" << per_shard_k << ") cannot exceed the smallest search_list value (" << min_L
           << ").";
        throw diskann::ANNException(ss.str(), -1);
    }

    const uint32_t search_threads = resolve_thread_count(config.search_threads);
    const uint32_t max_search_L = *std::max_element(search_lists.begin(), search_lists.end());

    fs::path output_root;
    if (!config.output_dir.empty())
    {
        output_root = fs::path(config.output_dir);
        fs::create_directories(output_root);
    }

    for (uint32_t shard_count : config.shard_counts)
    {
        fs::path shard_dir = fs::path(config.index_root) / ("S_" + std::to_string(shard_count));
        fs::path manifest_path = shard_dir / "manifest.json";
        if (!fs::exists(manifest_path))
        {
            std::cerr << "Manifest not found for shard count " << shard_count << " at " << manifest_path << std::endl;
            continue;
        }

        BuildManifest manifest = read_manifest(manifest_path);
        if (manifest.shard_count != shard_count)
        {
            std::cerr << "Warning: manifest shard_count " << manifest.shard_count
                      << " does not match requested shard count " << shard_count << std::endl;
        }
        if (manifest.data_type != config.data_type)
        {
            std::cerr << "Warning: manifest data_type is '" << manifest.data_type << "' but search requested '"
                      << config.data_type << "'." << std::endl;
        }

        std::cout << "\nEvaluating shards for S=" << shard_count << " from " << shard_dir << std::endl;

        std::vector<LoadedShard> shards = load_shards(manifest, shard_dir, metric, search_threads, max_search_L);

        struct ResultRow
        {
            uint32_t search_list;
            double recall;
            uint64_t total_comparisons;
        };
        std::vector<ResultRow> csv_rows;
        csv_rows.reserve(search_lists.size());

        for (uint32_t L : search_lists)
        {
            auto summary = evaluate_shards(shards, metric, queries, query_count, query_aligned_dim, config.recall_at,
                                           per_shard_k, L, gt_ids, gt_dists, static_cast<uint32_t>(gt_dim),
                                           has_truth, search_threads);

            std::cout << "S=" << shard_count << " L=" << L;
            if (has_truth)
            {
                std::cout << " Recall@" << config.recall_at << ": " << summary.recall << "%";
            }
            std::cout << " Avg Comparisons/query: " << summary.avg_comparisons
                      << " Total Comparisons: " << summary.total_comparisons << std::endl;

            csv_rows.push_back(ResultRow{L, summary.recall, summary.total_comparisons});
        }

        if (!config.output_dir.empty())
        {
            fs::path csv_path = output_root / (std::string("S_") + std::to_string(shard_count) + ".csv");
            std::ofstream csv(csv_path, std::ios::trunc);
            if (!csv.is_open())
            {
                std::stringstream ss;
                ss << "Failed to open CSV output file " << csv_path;
                throw diskann::ANNException(ss.str(), -1);
            }

            csv << "search_list,recall,total_comparisons\n";
            csv.setf(std::ios::fixed, std::ios::floatfield);
            csv.precision(6);
            for (const auto &row : csv_rows)
            {
                csv << row.search_list << ',';
                if (std::isfinite(row.recall))
                {
                    csv << row.recall;
                }
                else
                {
                    csv << "NaN";
                }
                csv << ',' << row.total_comparisons << '\n';
            }
            csv.close();

            std::cout << "CSV summary written to " << csv_path << std::endl;
        }
    }

    diskann::aligned_free(queries);
    if (gt_ids != nullptr)
    {
        delete[] gt_ids;
    }
    if (gt_dists != nullptr)
    {
        delete[] gt_dists;
    }

    return 0;
}

int main(int argc, char **argv)
{
    SearchConfig config = parse_arguments(argc, argv);

    try
    {
        if (config.data_type == std::string("float"))
        {
            return run_search<float>(config);
        }
        if (config.data_type == std::string("uint8"))
        {
            return run_search<uint8_t>(config);
        }
        if (config.data_type == std::string("int8"))
        {
            return run_search<int8_t>(config);
        }
        std::stringstream ss;
        ss << "Unsupported data_type '" << config.data_type << "'. Supported: float, uint8, int8.";
        throw diskann::ANNException(ss.str(), -1);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }
}
