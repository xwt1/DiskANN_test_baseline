// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include <boost/program_options.hpp>

#include "ann_exception.h"
#include "compare_common.hpp"
#include "index_factory.h"
#include "program_options_utils.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;
using namespace compare_experiment;

struct BuildConfig
{
    std::string data_type;
    std::string dist_fn;
    std::string data_path;
    uint32_t shard_count = 1;
    std::string output_root;
    uint32_t build_L = 100;
    uint32_t build_R = 64;
    float build_alpha = 1.2f;
    uint32_t build_threads = 0;
    std::optional<uint32_t> random_seed;
};

template <typename T>
std::unique_ptr<diskann::AbstractIndex> build_index_for_data(const BuildConfig &config, diskann::Metric metric,
                                                             const T *data, size_t points, size_t dim,
                                                             diskann::BuildComputeStats &stats)
{
    using TagT = uint32_t;
    using LabelT = uint32_t;

    const uint32_t configured_threads =
        config.build_threads == 0 ? static_cast<uint32_t>(omp_get_num_procs()) : config.build_threads;
    const uint32_t build_threads = std::max<uint32_t>(1, configured_threads);

    auto build_params = diskann::IndexWriteParametersBuilder(config.build_L, config.build_R)
                            .with_alpha(config.build_alpha)
                            .with_num_threads(build_threads)
                            .with_saturate_graph(false)
                            .build();

    auto index_config = diskann::IndexConfigBuilder()
                            .with_metric(metric)
                            .with_dimension(dim)
                            .with_max_points(points)
                            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                            .with_data_type(type_name<T>())
                            .with_label_type(type_name<LabelT>())
                            .with_tag_type(type_name<TagT>())
                            .with_index_write_params(build_params)
                            .is_dynamic_index(false)
                            .is_enable_tags(false)
                            .is_concurrent_consolidate(false)
                            .is_pq_dist_build(false)
                            .is_use_opq(false)
                            .build();

    diskann::IndexFactory factory(index_config);
    auto index = factory.create_instance();
    std::vector<TagT> tags;
    index->reset_build_compute_stats();
    index->build(data, points, tags);
    stats = index->get_build_compute_stats();
    index->reset_build_compute_stats();
    index->reset_search_compute_stats();
    return index;
}

std::vector<size_t> compute_shard_sizes(size_t total_points, uint32_t shard_count)
{
    if (shard_count == 0)
    {
        throw diskann::ANNException("Number of shards must be positive.", -1);
    }
    std::vector<size_t> sizes(shard_count);
    const size_t base = total_points / shard_count;
    const size_t remainder = total_points % shard_count;
    for (uint32_t i = 0; i < shard_count; ++i)
    {
        sizes[i] = base + (i < remainder ? 1 : 0);
    }
    return sizes;
}

template <typename T> int run_build(const BuildConfig &config)
{
    diskann::Metric metric = parse_metric(config.dist_fn, config.data_type);

    T *raw = nullptr;
    size_t total_points = 0;
    size_t dim = 0;
    diskann::load_bin<T>(config.data_path, raw, total_points, dim);
    std::unique_ptr<T[]> data_guard(raw);

    if (total_points == 0)
    {
        throw diskann::ANNException("Dataset is empty.", -1);
    }
    if (config.shard_count > total_points)
    {
        std::stringstream ss;
        ss << "Shard count (" << config.shard_count << ") exceeds number of data points (" << total_points << ").";
        throw diskann::ANNException(ss.str(), -1);
    }

    const uint32_t seed = config.random_seed.has_value() ? config.random_seed.value() : std::random_device{}();
    std::vector<uint32_t> permutation(total_points);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(permutation.begin(), permutation.end(), rng);

    const std::vector<size_t> shard_sizes = compute_shard_sizes(total_points, config.shard_count);

    fs::path root = fs::path(config.output_root);
    fs::create_directories(root);
    fs::path shard_dir = root / ("S_" + std::to_string(config.shard_count));
    fs::create_directories(shard_dir);

    std::cout << "Building " << config.shard_count << " shard indexes under " << shard_dir << std::endl;

    std::vector<BuildManifestShard> manifest_shards;
    manifest_shards.reserve(config.shard_count);

    size_t offset = 0;
    for (uint32_t shard_idx = 0; shard_idx < config.shard_count; ++shard_idx)
    {
        const size_t shard_size = shard_sizes[shard_idx];
        std::vector<T> shard_data(shard_size * dim);
        std::vector<uint32_t> mapping(shard_size);

        for (size_t i = 0; i < shard_size; ++i)
        {
            const uint32_t original = permutation[offset + i];
            mapping[i] = original;
            const T *src = raw + static_cast<size_t>(original) * dim;
            std::copy(src, src + dim, shard_data.data() + i * dim);
        }

        diskann::BuildComputeStats stats;
        auto index = build_index_for_data(config, metric, shard_data.data(), shard_size, dim, stats);

        std::ostringstream index_name;
        index_name << "shard_" << std::setw(3) << std::setfill('0') << shard_idx << ".index";
        std::ostringstream map_name;
        map_name << "shard_" << std::setw(3) << std::setfill('0') << shard_idx << ".map";

        const fs::path index_path = shard_dir / index_name.str();
        const fs::path map_path = shard_dir / map_name.str();

        index->save(index_path.string().c_str());
        save_mapping(map_path, mapping);

        manifest_shards.push_back(BuildManifestShard{shard_idx, shard_size, index_name.str(), map_name.str(), stats});

        std::cout << "Shard " << shard_idx << ": built index over " << shard_size
                  << " points (distance comparisons: " << stats.total_distance_comparisons << ")." << std::endl;

        offset += shard_size;
    }

    BuildManifest manifest;
    manifest.data_type = config.data_type;
    manifest.dist_fn = config.dist_fn;
    manifest.metric = metric;
    manifest.dimension = dim;
    manifest.total_points = total_points;
    manifest.shard_count = config.shard_count;
    manifest.build_L = config.build_L;
    manifest.build_R = config.build_R;
    manifest.build_alpha = config.build_alpha;
    manifest.build_threads = config.build_threads == 0 ? static_cast<uint32_t>(omp_get_num_procs()) : config.build_threads;
    manifest.random_seed = seed;
    manifest.shards = std::move(manifest_shards);

    const fs::path manifest_path = shard_dir / "manifest.json";
    write_manifest(manifest, manifest_path);

    std::cout << "Manifest written to " << manifest_path << std::endl;
    return 0;
}

BuildConfig parse_arguments(int argc, char **argv)
{
    BuildConfig config;

    po::options_description desc{
        program_options_utils::make_program_description("compare_build", "Shard-aware index construction")};

    try
    {
        po::options_description required("Required");
        required.add_options()
            ("data_type", po::value<std::string>(&config.data_type)->required(),
             program_options_utils::DATA_TYPE_DESCRIPTION)
            ("dist_fn", po::value<std::string>(&config.dist_fn)->required(),
             program_options_utils::DISTANCE_FUNCTION_DESCRIPTION)
            ("data_path", po::value<std::string>(&config.data_path)->required(),
             program_options_utils::INPUT_DATA_PATH)
            ("output_root", po::value<std::string>(&config.output_root)->required(),
             "Root directory where shard folders will be written")
            ("shard_count", po::value<uint32_t>(&config.shard_count)->required(),
             "Number of random shards to construct");

        po::options_description optional("Optional");
        optional.add_options()
            ("Lbuild", po::value<uint32_t>(&config.build_L)->default_value(100),
             program_options_utils::GRAPH_BUILD_COMPLEXITY)
            ("max_degree", po::value<uint32_t>(&config.build_R)->default_value(64),
             program_options_utils::MAX_BUILD_DEGREE)
            ("alpha", po::value<float>(&config.build_alpha)->default_value(1.2f),
             program_options_utils::GRAPH_BUILD_ALPHA)
            ("build_threads", po::value<uint32_t>(&config.build_threads)->default_value(0),
             program_options_utils::NUMBER_THREADS_DESCRIPTION)
            ("random_seed", po::value<uint32_t>(), "Random seed for sharding permutation");

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

        if (vm.count("random_seed"))
        {
            config.random_seed = vm["random_seed"].as<uint32_t>();
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        std::cerr << desc << std::endl;
        std::exit(-1);
    }

    return config;
}

int main(int argc, char **argv)
{
    BuildConfig config = parse_arguments(argc, argv);

    try
    {
        if (config.data_type == std::string("float"))
        {
            return run_build<float>(config);
        }
        if (config.data_type == std::string("uint8"))
        {
            return run_build<uint8_t>(config);
        }
        if (config.data_type == std::string("int8"))
        {
            return run_build<int8_t>(config);
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
