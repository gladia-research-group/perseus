#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace perseus_np {

namespace py = pybind11;
using Arr1 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Arr2 = py::array_t<double, py::array::c_style | py::array::forcecast>;

inline std::vector<double> to_vec(const Arr1& a) {
    if (a.ndim() != 1)
        throw std::invalid_argument("expected a 1-D array, got " + std::to_string(a.ndim()) + "-D");
    const auto r = a.unchecked<1>();
    std::vector<double> v(static_cast<size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) v[static_cast<size_t>(i)] = r(i);
    return v;
}

inline std::vector<std::vector<double>> to_mat(const Arr2& a) {
    if (a.ndim() != 2)
        throw std::invalid_argument("expected a 2-D array, got " + std::to_string(a.ndim()) + "-D");
    const auto r = a.unchecked<2>();
    std::vector<std::vector<double>> m(static_cast<size_t>(r.shape(0)),
                                       std::vector<double>(static_cast<size_t>(r.shape(1))));
    for (py::ssize_t i = 0; i < r.shape(0); ++i)
        for (py::ssize_t j = 0; j < r.shape(1); ++j)
            m[static_cast<size_t>(i)][static_cast<size_t>(j)] = r(i, j);
    return m;
}

inline py::array_t<double> from_vec(const std::vector<double>& v) {
    py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
    auto w = out.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i) w(static_cast<py::ssize_t>(i)) = v[i];
    return out;
}

inline py::array_t<double> from_mat(const std::vector<std::vector<double>>& m) {
    const py::ssize_t rows = static_cast<py::ssize_t>(m.size());
    const py::ssize_t cols = rows ? static_cast<py::ssize_t>(m[0].size()) : 0;
    py::array_t<double> out({rows, cols});
    auto w = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < rows; ++i)
        for (py::ssize_t j = 0; j < cols; ++j)
            w(i, j) = m[static_cast<size_t>(i)][static_cast<size_t>(j)];
    return out;
}

}  // namespace perseus_np
