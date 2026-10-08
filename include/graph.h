#pragma once

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct GraphNode {
    std::size_t id = 0;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<int> input_levels;
    std::string output;
    int output_level = -1;
    bool has_output_noise_level = false;
    int output_noise_level = -1;
    bool has_output_max_abs = false;
    double output_max_abs = 0.0;
    bool has_output_stats = false;
    double output_mean = 0.0;
    double output_max_dev = 0.0;
    // max plaintext coefficient of the output (what EvalMod sees); _ac excludes the X^0 term
    bool has_output_max_coeff = false;
    double output_max_coeff = 0.0;
    double output_max_coeff_ac = 0.0;
    std::string step;
    bool has_pack_tag = false;
    int  pack_period  = 0;
    int  pack_kind    = 0;   // packtag::Support::Kind: 0=Empty, 1=AP, 2=Dense
    int  pack_offset  = 0;
    int  pack_stride  = 0;
    int  pack_count   = 0;
    int  pack_width   = 1;   // Block support width (1 = classic AP)

    bool   has_fold       = false;
    double fold_stride    = 0.0;
    double recovery_const = 0.0;
};

class ComputationGraph {
public:
    std::size_t add_node(std::string op_type,
                         std::vector<std::string> inputs,
                         std::string output,
                         std::string step = "") {
        std::vector<int> input_levels(inputs.size(), -1);
        return add_node(std::move(op_type), std::move(inputs), std::move(output),
                        std::move(input_levels), -1, false, -1, false, 0.0,
                        std::move(step));
    }

    std::size_t add_node(std::string op_type,
                         std::vector<std::string> inputs,
                         std::string output,
                         std::vector<int> input_levels,
                         int output_level,
                         bool has_output_noise_level,
                         int output_noise_level,
                         bool has_output_max_abs,
                         double output_max_abs,
                         std::string step = "") {
        GraphNode node;
        node.op_type = std::move(op_type);
        node.inputs = std::move(inputs);
        node.output = std::move(output);
        node.input_levels = std::move(input_levels);
        node.output_level = output_level;
        node.has_output_noise_level = has_output_noise_level;
        node.output_noise_level = output_noise_level;
        node.has_output_max_abs = has_output_max_abs;
        node.output_max_abs = output_max_abs;
        node.step = std::move(step);
        nodes_.push_back(std::move(node));
        return nodes_.back().id;
    }

    const std::vector<GraphNode>& nodes() const {
        return nodes_;
    }

    GraphNode* last_node() {
        return nodes_.empty() ? nullptr : &nodes_.back();
    }

    // The latest node that produced `output` (nullptr if none).
    GraphNode* last_node_for(const std::string& output) {
        for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it)
            if (it->output == output) return &*it;
        return nullptr;
    }

    std::size_t size() const { return nodes_.size(); }

    void set_output_stats(std::size_t idx, double mean, double max_dev) {
        if (idx < nodes_.size()) {
            nodes_[idx].has_output_stats = true;
            nodes_[idx].output_mean = mean;
            nodes_[idx].output_max_dev = max_dev;
        }
    }

    void set_output_max_coeff(std::size_t idx, double max_coeff, double max_coeff_ac) {
        if (idx < nodes_.size()) {
            nodes_[idx].has_output_max_coeff = true;
            nodes_[idx].output_max_coeff = max_coeff;
            nodes_[idx].output_max_coeff_ac = max_coeff_ac;
        }
    }
    void set_output_max_abs(std::size_t idx, double v) {
        if (idx < nodes_.size()) {
            nodes_[idx].has_output_max_abs = true;
            nodes_[idx].output_max_abs = v;
        }
    }

    void clear() {
        nodes_.clear();
    }

    std::string to_json() const {
        std::ostringstream out;
        out << "{\n";
        out << "  \"version\": 1,\n";
        out << "  \"nodes\": [\n";
        for (std::size_t i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            out << "    {\n";
            out << "      \"op_type\": \"" << escape_json(node.op_type) << "\",\n";
            out << "      \"inputs\": [";
            for (std::size_t j = 0; j < node.inputs.size(); ++j) {
                if (j > 0) {
                    out << ", ";
                }
                out << "\"" << escape_json(node.inputs[j]) << "\"";
            }
            out << "],\n";
            out << "      \"input_levels\": [";
            for (std::size_t j = 0; j < node.input_levels.size(); ++j) {
                if (j > 0) {
                    out << ", ";
                }
                out << node.input_levels[j];
            }
            out << "],\n";
            out << "      \"output\": \"" << escape_json(node.output) << "\",\n";
            out << "      \"step\": \"" << escape_json(node.step) << "\",\n";
            out << "      \"output_level\": " << node.output_level;
            if (node.has_output_noise_level) {
                out << ",\n";
                out << "      \"output_noise_level\": " << node.output_noise_level;
            }
            if (node.has_output_max_abs) {
                out << ",\n";
                out << "      \"output_max_abs\": "
                    << std::setprecision(17) << node.output_max_abs;
            }
            if (node.has_output_stats) {
                out << ",\n";
                out << "      \"output_mean\": "
                    << std::setprecision(17) << node.output_mean << ",\n";
                out << "      \"output_max_dev\": "
                    << std::setprecision(17) << node.output_max_dev;
            }
            if (node.has_output_max_coeff) {
                out << ",\n";
                out << "      \"output_max_coeff\": "
                    << std::setprecision(17) << node.output_max_coeff << ",\n";
                out << "      \"output_max_coeff_ac\": "
                    << std::setprecision(17) << node.output_max_coeff_ac;
            }
            if (node.has_pack_tag) {
                out << ",\n";
                out << "      \"pack_period\": " << node.pack_period << ",\n";
                out << "      \"pack_kind\": " << node.pack_kind << ",\n";
                out << "      \"pack_offset\": " << node.pack_offset << ",\n";
                out << "      \"pack_stride\": " << node.pack_stride << ",\n";
                out << "      \"pack_count\": " << node.pack_count << ",\n";
                out << "      \"pack_width\": " << node.pack_width;
            }
            if (node.has_fold) {
                out << ",\n";
                out << "      \"fold_stride\": "
                    << std::setprecision(17) << node.fold_stride << ",\n";
                out << "      \"recovery_const\": "
                    << std::setprecision(17) << node.recovery_const;
            }
            out << "\n";
            out << "    }";
            if (i + 1 < nodes_.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";
        return out.str();
    }

private:
    static std::string escape_json(const std::string& text) {
        std::ostringstream out;
        for (char ch : text) {
            switch (ch) {
                case '\\': out << "\\\\"; break;
                case '"': out << "\\\""; break;
                case '\b': out << "\\b"; break;
                case '\f': out << "\\f"; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default: out << ch; break;
            }
        }
        return out.str();
    }

    std::vector<GraphNode> nodes_;
};

class GraphBuilder {
public:
    GraphBuilder() : graph_(std::make_shared<ComputationGraph>()) {}

    explicit GraphBuilder(std::shared_ptr<ComputationGraph> graph)
        : graph_(std::move(graph)) {
        if (!graph_) {
            graph_ = std::make_shared<ComputationGraph>();
        }
    }

    bool enabled() const {
        return static_cast<bool>(graph_);
    }

    std::size_t add_node(const std::string& op_type,
                         std::initializer_list<std::string> inputs,
                         const std::string& output,
                         const std::string& step = "") {
        return graph_->add_node(op_type, std::vector<std::string>(inputs), output, step);
    }

    std::size_t add_node(const std::string& op_type,
                         std::initializer_list<std::string> inputs,
                         const std::string& output,
                 const std::vector<int>& input_levels,
                 int output_level,
                 bool has_output_noise_level,
                 int output_noise_level,
                 bool has_output_max_abs,
                 double output_max_abs,
                 const std::string& step = "") {
        return graph_->add_node(op_type,
                                std::vector<std::string>(inputs),
                                output,
                    input_levels,
                    output_level,
                    has_output_noise_level,
                    output_noise_level,
                    has_output_max_abs,
                    output_max_abs,
                    step);
    }

    std::size_t add_node(const std::string& op_type,
                         const std::vector<std::string>& inputs,
                         const std::string& output,
                         const std::string& step = "") {
        return graph_->add_node(op_type, inputs, output, step);
    }

    std::size_t add_node(const std::string& op_type,
                         const std::vector<std::string>& inputs,
                         const std::string& output,
                         const std::vector<int>& input_levels,
                         int output_level,
                    bool has_output_noise_level,
                    int output_noise_level,
                         bool has_output_max_abs,
                         double output_max_abs,
                         const std::string& step = "") {
        return graph_->add_node(op_type, inputs, output, input_levels, output_level,
                        has_output_noise_level, output_noise_level,
                        has_output_max_abs, output_max_abs, step);
    }

    GraphNode* last_node() {
        return graph_ ? graph_->last_node() : nullptr;
    }

    GraphNode* last_node_for(const std::string& output) {
        return graph_ ? graph_->last_node_for(output) : nullptr;
    }

    std::string to_json() const {
        return graph_->to_json();
    }

    void clear() {
        graph_->clear();
    }

    void export_json(const std::string& path) const {
        std::ofstream out(path);
        out << to_json();
    }

    std::shared_ptr<ComputationGraph> graph() const {
        return graph_;
    }

private:
    std::shared_ptr<ComputationGraph> graph_;
};