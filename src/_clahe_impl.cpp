#include <pybind11/numpy.h>

// Adaptive Histogram Equalization and Its Variations, Pizer et al. (1987).

namespace py = pybind11;
using namespace pybind11::literals;
using ssize_t = py::ssize_t;

template<typename T>
py::array_t<double> clahe(
  py::array_t<T> img, ssize_t wi, ssize_t wj, ssize_t wk,
  double clip_limit, bool multiplicative)
{
  auto np = py::module::import("numpy");
  auto py_orig_vals = py::object{np.attr("unique")(img)}.cast<py::array>();
  auto orig_vals = py_orig_vals.unchecked<T, 1>();
  auto py_img_ord = py::object{np.attr("searchsorted")(py_orig_vals, img)}.cast<py::array>();
  auto img_ord = py_img_ord.unchecked<size_t, 3>();
  size_t ni = img_ord.shape(0), nj = img_ord.shape(1), nk = img_ord.shape(2),
         hwi = (wi - 1) / 2, hwj = (wj - 1) / 2, hwk = (wk - 1) / 2,
         win_size = wi * wj * wk,
         nvals = orig_vals.size();
  double count_clip = clip_limit * win_size / nvals;
  auto hist = std::unique_ptr<size_t[]>{new size_t[nvals]};
  auto py_out = py::array_t<double>{{ni, nj, nk}};
  auto out = py_out.mutable_unchecked<3>();
  for (auto k0 = 0u; k0 < nk - wk + 1; ++k0) {
    for (auto j0 = 0u; j0 < nj - wj + 1; ++j0) {
      std::fill_n(hist.get(), nvals, 0);
      for (auto k = k0; k < k0 + wk; ++k) {
        for (auto j = j0; j < j0 + wj; ++j) {
          for (auto i = 0u; i < wi - 1; ++i) {
            ++hist[img_ord(i, j, k)];
          }
        }
      }
      for (auto i0 = 0u; i0 < ni - wi + 1; ++i0) {
        // Update histogram.
        for (auto k = k0; k < k0 + wk; ++k) {
          for (auto j = j0; j < j0 + wj; ++j) {
            ++hist[img_ord(i0 + wi - 1, j, k)];
          }
        }
        // Limit contrast.
        auto val = img_ord(i0 + hwi, j0 + hwj, k0 + hwk);
        auto clip_sum = 0.;
        for (auto viter = 0u; viter < val; ++viter) {
          clip_sum += std::min<double>(hist[viter], count_clip);
        }
        auto clip_psum =
          clip_sum + std::min<double>(hist[val], count_clip) / 2.;
        for (auto viter = val; viter < nvals; ++viter) {
          clip_sum += std::min<double>(hist[viter], count_clip);
        }
        out(i0 + hwi, j0 + hwj, k0 + hwk) =
          multiplicative
          ? clip_psum / clip_sum
          : ((clip_psum
              + (win_size - clip_sum)
              * (orig_vals(val) + .5) / (orig_vals(nvals - 1) + 1))
             / win_size);
        // Update histogram.
        for (auto k = k0; k < k0 + wk; ++k) {
          for (auto j = j0; j < j0 + wj; ++j) {
            --hist[img_ord(i0, j, k)];
          }
        }
      }
    }
  }
  return py_out;
}

template<typename T, typename M>
void declare_api(M module) {
  // img is noconvert, otherwise the first overload will always be used.
  module.def("clahe", clahe<T>,
             "img"_a.noconvert(), "wi"_a, "wj"_a, "wk"_a,
             "clip_limit"_a, "multiplicative"_a=false);
}

PYBIND11_MODULE(_clahe_impl, m) {
  declare_api<uint8_t>(m);
  declare_api<int8_t>(m);
  declare_api<uint16_t>(m);
  declare_api<int16_t>(m);
  declare_api<uint32_t>(m);
  declare_api<int32_t>(m);
  declare_api<uint64_t>(m);
  declare_api<int64_t>(m);
}
