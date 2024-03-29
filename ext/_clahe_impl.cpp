#include <pybind11/numpy.h>
#include <limits>

// Adaptive Histogram Equalization and Its Variations, Pizer et al. (1987).

namespace py = pybind11;
using namespace pybind11::literals;
using ssize_t = py::ssize_t;

template<typename Data, typename Accum>
py::array_t<double> clahe(
  py::array_t<Data> img, ssize_t wi, ssize_t wj, ssize_t wk,
  double clip_limit, bool multiplicative)
{
  auto np = py::module::import("numpy");
  auto py_orig_vals = py::object{np.attr("unique")(img)}.cast<py::array>();
  auto orig_vals = py_orig_vals.unchecked<Data, 1>();
  auto py_img_ord =
    py::object{np.attr("searchsorted")(py_orig_vals, img)}.cast<py::array>();
  auto img_ord = py_img_ord.unchecked<size_t, 3>();
  size_t ni = img_ord.shape(0), nj = img_ord.shape(1), nk = img_ord.shape(2),
         hwi = (wi - 1) / 2, hwj = (wj - 1) / 2, hwk = (wk - 1) / 2,
         win_size = wi * wj * wk,
         nvals = orig_vals.size();
  double count_clip = clip_limit * win_size / nvals;
  Accum count_iclip = Accum(count_clip);
  auto hist = std::unique_ptr<Accum[]>{new Accum[nvals]};
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
        auto clip_isum = Accum(0), n_clip = Accum(0);
        for (auto viter = 0u; viter < val; ++viter) {
          // Avoiding fp comparisons and keeping int sums/accumulating all the
          // clips at once later is faster.
          if (hist[viter] > count_iclip) {
            ++n_clip;
          } else {
            clip_isum += hist[viter];
          }
        }
        auto clip_psum = clip_isum + n_clip * count_clip
                         + std::min<double>(hist[val], count_clip) / 2.;
        for (auto viter = val; viter < nvals; ++viter) {
          if (hist[viter] > count_iclip) {
            ++n_clip;
          } else {
            clip_isum += hist[viter];
          }
        }
        auto clip_sum = clip_isum + n_clip * count_clip;
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

template<typename Data>
py::array_t<double> clahe_dispatch(
  py::array_t<Data> img, ssize_t wi, ssize_t wj, ssize_t wk,
  double clip_limit, bool multiplicative)
{
  // Using a smaller accumulator type is faster.
  auto win_size = uintmax_t(wi) * uintmax_t(wj) * uintmax_t(wk);
  return
    (win_size <= std::numeric_limits<uint8_t>::max() ? clahe<Data, uint8_t> :
     win_size <= std::numeric_limits<uint16_t>::max() ? clahe<Data, uint16_t> :
     win_size <= std::numeric_limits<uint32_t>::max() ? clahe<Data, uint32_t> :
     clahe<Data, uint64_t>)(
       img, wi, wj, wk, clip_limit, multiplicative);
}

template<typename Data, typename Module>
void declare_api(Module module) {
  // img is noconvert, otherwise the first overload will always be used.
  module.def("clahe", clahe_dispatch<Data>,
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
