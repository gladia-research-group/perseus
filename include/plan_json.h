#pragma once

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace planjson {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;   // insertion order preserved

    bool is_object() const { return type == Type::Object; }
    bool is_array()  const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }
    bool is_number() const { return type == Type::Number; }

    const Value* find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

inline Value parse(const std::string& s) {
    size_t i = 0;
    const size_t n = s.size();

    auto fail = [&](const char* what) -> void {
        throw std::runtime_error("planjson: " + std::string(what) +
                                 " at offset " + std::to_string(i));
    };
    auto skip_ws = [&]() {
        while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
            ++i;
    };
    auto parse_string = [&]() -> std::string {
        ++i;
        std::string out;
        while (i < n && s[i] != '"') {
            char c = s[i];
            if (c == '\\') {
                if (i + 1 >= n) fail("dangling escape");
                char e = s[++i];
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u':
                        if (i + 4 >= n) fail("truncated \\u escape");
                        out += "\\u";
                        out.append(s, i + 1, 4);
                        i += 4;
                        break;
                    default: fail("unknown escape");
                }
            } else {
                out += c;
            }
            ++i;
        }
        if (i >= n) fail("unterminated string");
        ++i;
        return out;
    };
    auto parse_scalar = [&]() -> Value {
        Value v;
        char c = s[i];
        if (c == '"') {
            v.type = Value::Type::String;
            v.str = parse_string();
        } else if (c == 't' && s.compare(i, 4, "true") == 0) {
            v.type = Value::Type::Bool; v.boolean = true; i += 4;
        } else if (c == 'f' && s.compare(i, 5, "false") == 0) {
            v.type = Value::Type::Bool; v.boolean = false; i += 5;
        } else if (c == 'n' && s.compare(i, 4, "null") == 0) {
            v.type = Value::Type::Null; i += 4;
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            const char* start = s.c_str() + i;
            char* end = nullptr;
            v.type = Value::Type::Number;
            v.number = std::strtod(start, &end);
            if (end == start) fail("bad number");
            i += static_cast<size_t>(end - start);
        } else {
            fail("unexpected character");
        }
        return v;
    };

    struct Frame {
        Value v;
        std::string pending_key;
        bool has_key = false;
    };
    std::vector<Frame> stack;
    Value root;
    bool have_root = false;

    auto attach = [&](Value&& v) {
        if (stack.empty()) {
            if (have_root) fail("multiple top-level values");
            root = std::move(v);
            have_root = true;
            return;
        }
        Frame& top = stack.back();
        if (top.v.type == Value::Type::Object) {
            if (!top.has_key) fail("object value without key");
            top.v.obj.emplace_back(std::move(top.pending_key), std::move(v));
            top.has_key = false;
            top.pending_key.clear();
        } else {
            top.v.arr.push_back(std::move(v));
        }
    };

    while (true) {
        skip_ws();
        if (i >= n) break;
        char c = s[i];
        if (c == '{') {
            Frame f; f.v.type = Value::Type::Object;
            stack.push_back(std::move(f));
            ++i;
        } else if (c == '[') {
            Frame f; f.v.type = Value::Type::Array;
            stack.push_back(std::move(f));
            ++i;
        } else if (c == '}' || c == ']') {
            if (stack.empty()) fail("unmatched close");
            if ((c == '}') != (stack.back().v.type == Value::Type::Object))
                fail("mismatched close");
            Value done = std::move(stack.back().v);
            stack.pop_back();
            attach(std::move(done));
            ++i;
        } else if (c == ',' || c == ':') {
            ++i;
        } else if (c == '"' && !stack.empty()
                   && stack.back().v.type == Value::Type::Object
                   && !stack.back().has_key) {
            stack.back().pending_key = parse_string();
            stack.back().has_key = true;
        } else {
            attach(parse_scalar());
        }
        if (have_root && stack.empty()) break;
    }
    if (!have_root || !stack.empty())
        throw std::runtime_error("planjson: truncated document");
    return root;
}

}   // namespace planjson
