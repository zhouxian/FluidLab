#pragma once

#include <vector>

#include "core.h"
#include "maths.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct SmokeParticles
{
    int particles_size;
    std::vector<Point3> p_positions;
    std::vector<Colour> p_colours;
    std::vector<int> p_indices;
    py::array_t<float> p_positions_raw;
    py::array_t<float> p_colours_raw;
};