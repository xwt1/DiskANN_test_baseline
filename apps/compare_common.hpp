#pragma once

#include <algorithm>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "ann_exception.h"
#include "common_includes.h"
#include "logger.h"
#include "abstract_index.h"
#include "distance.h"
#include "index_build_params.h"
#include "index_config.h"
#include "utils.h"

namespace compare_experiment
{
namespace fs = std::filesystem;
namespace pt = boost::property_tree;

struct BuildManifestShard
{
    uint32_t ordinal = 0;
    size_t point_count = 0;
    std::string index_file;
    std::string map_file;
    diskann::BuildComputeStats stats{};
};

struct BuildManifest
{
    std::string data_type;
    std::string dist_fn;
    diskann::Metric metric = diskann::Metric::L2;
    size_t dimension = 0;
    size_t total_points = 0;
    uint32_t shard_count = 1;
    uint32_t build_L = 0;
    uint32_t build_R = 0;
    float build_alpha = 0.0f;
    uint32_t build_threads = 0;
    uint32_t random_seed = 0;
    std::vector<BuildManifestShard> shards;
};

template <typename T> inline std::string type_name();

template <> inline std::string type_name<float>() { return "float"; }
template <> inline std::string type_name<uint8_t>() { return "uint8"; }
template <> inline std::string type_name<int8_t>() { return "int8"; }
template <> inline std::string type_name<uint32_t>() { return "uint32"; }

enum class DistancePreference
{
    Minimize,
    Maximize
};

inline bool is_null_token(const std::string &token)
{
    return token.empty() || token == "null" || token == "NULL";
}

inline std::string metric_to_string(diskann::Metric metric)
{
    switch (metric)
    {
    case diskann::Metric::L2:
        return "l2";
    case diskann::Metric::COSINE:
        return "cosine";
    case diskann::Metric::INNER_PRODUCT:
        return "mips";
    case diskann::Metric::FAST_L2:
        return "fast_l2";
    default:
        return "unknown";
    }
}

inline diskann::Metric parse_metric(const std::string &dist_fn, const std::string &data_type)
{
    if ((dist_fn == std::string("mips")) && (data_type == std::string("float")))
    {
        return diskann::Metric::INNER_PRODUCT;
    }
    if (dist_fn == std::string("fast_l2") && data_type == std::string("float"))
    {
        return diskann::Metric::FAST_L2;
    }
    if (dist_fn == std::string("l2"))
    {
        return diskann::Metric::L2;
    }
    if (dist_fn == std::string("cosine"))
    {
        return diskann::Metric::COSINE;
    }

    std::stringstream ss;
    ss << "Unsupported distance function '" << dist_fn << "' for data type '" << data_type
       << "'. Supported options: l2, cosine, mips/fast_l2 (float only).";
    throw diskann::ANNException(ss.str(), -1);
}

inline DistancePreference preference_for_metric(diskann::Metric metric)
{
    switch (metric)
    {
    case diskann::Metric::INNER_PRODUCT:
        return DistancePreference::Maximize;
    default:
        return DistancePreference::Minimize;
    }
}

inline void write_manifest(const BuildManifest &manifest, const fs::path &path)
{
    pt::ptree root;
    root.put("data_type", manifest.data_type);
    root.put("dist_fn", manifest.dist_fn);
    root.put("metric", metric_to_string(manifest.metric));
    root.put("dimension", manifest.dimension);
    root.put("total_points", manifest.total_points);
    root.put("shard_count", manifest.shard_count);
    root.put("build_params.L", manifest.build_L);
    root.put("build_params.R", manifest.build_R);
    root.put("build_params.alpha", manifest.build_alpha);
    root.put("build_params.threads", manifest.build_threads);
    root.put("random_seed", manifest.random_seed);

    pt::ptree shards_node;
    for (const auto &shard : manifest.shards)
    {
        pt::ptree node;
        node.put("index", shard.ordinal);
        node.put("size", shard.point_count);
        node.put("index_file", shard.index_file);
        node.put("map_file", shard.map_file);
        node.put("build_stats.total_invocations", shard.stats.total_invocations);
        node.put("build_stats.total_hops", shard.stats.total_hops);
        node.put("build_stats.total_distance_comparisons", shard.stats.total_distance_comparisons);
        shards_node.push_back(std::make_pair("", node));
    }

    root.add_child("shards", shards_node);
    pt::write_json(path.string(), root);
}

inline BuildManifest read_manifest(const fs::path &path)
{
    pt::ptree root;
    pt::read_json(path.string(), root);

    BuildManifest manifest;
    manifest.data_type = root.get<std::string>("data_type");
    manifest.dist_fn = root.get<std::string>("dist_fn", root.get<std::string>("metric"));
    manifest.metric = parse_metric(root.get<std::string>("metric", manifest.dist_fn), manifest.data_type);
    manifest.dimension = root.get<size_t>("dimension");
    manifest.total_points = root.get<size_t>("total_points");
    manifest.shard_count = root.get<uint32_t>("shard_count");
    manifest.build_L = root.get<uint32_t>("build_params.L", 0);
    manifest.build_R = root.get<uint32_t>("build_params.R", 0);
    manifest.build_alpha = root.get<float>("build_params.alpha", 0.0f);
    manifest.build_threads = root.get<uint32_t>("build_params.threads", 0);
    manifest.random_seed = root.get<uint32_t>("random_seed", 0);

    if (auto shards_opt = root.get_child_optional("shards"))
    {
        for (const auto &child : *shards_opt)
        {
            BuildManifestShard shard;
            shard.ordinal = child.second.get<uint32_t>("index");
            shard.point_count = child.second.get<size_t>("size");
            shard.index_file = child.second.get<std::string>("index_file");
            shard.map_file = child.second.get<std::string>("map_file");
            shard.stats.total_invocations = child.second.get<uint64_t>("build_stats.total_invocations", 0);
            shard.stats.total_hops = child.second.get<uint64_t>("build_stats.total_hops", 0);
            shard.stats.total_distance_comparisons =
                child.second.get<uint64_t>("build_stats.total_distance_comparisons", 0);
            manifest.shards.emplace_back(std::move(shard));
        }
    }

    return manifest;
}

inline void save_mapping(const fs::path &path, const std::vector<uint32_t> &mapping)
{
    if (mapping.empty())
    {
        uint32_t dummy = 0;
        diskann::save_bin<uint32_t>(path.string(), &dummy, 0, 1);
        return;
    }

    auto *raw_ptr = const_cast<uint32_t *>(mapping.data());
    diskann::save_bin<uint32_t>(path.string(), raw_ptr, mapping.size(), 1);
}

inline std::vector<uint32_t> load_mapping(const fs::path &path)
{
    std::unique_ptr<uint32_t[]> data;
    size_t npts = 0;
    size_t dim = 0;
    diskann::load_bin<uint32_t>(path.string(), data, npts, dim);
    if (dim != 1)
    {
        std::stringstream ss;
        ss << "Expected mapping file '" << path.string() << "' to have dimension 1, found " << dim << ".";
        throw diskann::ANNException(ss.str(), -1);
    }
    std::vector<uint32_t> mapping(npts);
    if (npts > 0)
    {
        std::copy_n(data.get(), npts, mapping.begin());
    }
    return mapping;
}

} // namespace compare_experiment
