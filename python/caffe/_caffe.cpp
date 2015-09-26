// pycaffe provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from Python.
// Note that for Python, we will simply use float as the data type.

#include <boost/make_shared.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// these need to be included after boost on OS X
#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <fstream>  // NOLINT

#include "_caffe.hpp"
#include "caffe/caffe.hpp"

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

namespace caffe {

// for convenience, check that input files can be opened, and raise an
// exception that boost will send to Python if not (caffe could still crash
// later if the input files are disturbed before they are actually used, but
// this saves frustration in most cases)
static void CheckFile(const string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
      f.close();
      throw std::runtime_error("Could not open file " + filename);
    }
    f.close();
}

bp::object PyBlobWrap::get_data() {
  npy_intp dims[] = {num(), channels(), height(), width()};

  PyObject *obj = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32,
                                            blob_->mutable_cpu_data());
  PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(obj), self_);
  Py_INCREF(self_);
  bp::handle<> h(obj);

  return bp::object(h);
}

bp::object PyBlobWrap::get_diff() {
  npy_intp dims[] = {num(), channels(), height(), width()};

  PyObject *obj = PyArray_SimpleNewFromData(4, dims, NPY_FLOAT32,
                                            blob_->mutable_cpu_diff());
  PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(obj), self_);
  Py_INCREF(self_);
  bp::handle<> h(obj);

  return bp::object(h);
}

PyNet::PyNet(string param_file, string pretrained_param_file) {
  Init(param_file);
  CheckFile(pretrained_param_file);
  net_->CopyTrainedLayersFrom(pretrained_param_file);
}

void PyNet::Init(string param_file) {
  CheckFile(param_file);
  net_.reset(new Net<float>(param_file));
}

void PyNet::check_contiguous_array(PyArrayObject* arr, string name,
    int channels, int height, int width) {
  if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS)) {
    throw std::runtime_error(name + " must be C contiguous");
  }
  if (PyArray_NDIM(arr) != 4) {
    throw std::runtime_error(name + " must be 4-d");
  }
  if (PyArray_TYPE(arr) != NPY_FLOAT32) {
    throw std::runtime_error(name + " must be float32");
  }
  if (PyArray_DIMS(arr)[1] != channels) {
    throw std::runtime_error(name + " has wrong number of channels");
  }
  if (PyArray_DIMS(arr)[2] != height) {
    throw std::runtime_error(name + " has wrong height");
  }
  if (PyArray_DIMS(arr)[3] != width) {
    throw std::runtime_error(name + " has wrong width");
  }
}

bp::list PyNet::InitializeImage(const string& imageName, int imageSize, bp::object meanImg, int cropsize) {
  // Upload mean image and source image to the GPU
  dev_mean_image_ = AllocateGpuBlob<float>(1,3,imageSize); // Expected size
  cropsize_ = cropsize;
  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(meanImg.ptr());
  CUDA_CHECK( cudaMemcpy(dev_mean_image_, PyArray_DATA(arr), sizeof(float) * imageSize * imageSize * 3, cudaMemcpyHostToDevice) );
  void* prt = LoadImageToGpuMat(imageName);
  dev_src_image_ = static_cast<cv::gpu::GpuMat*>(prt);
  //void* ptt = LoadImageToGpuMat(imageName);
  //dev_stt_image_ = static_cast<cv::gpu::GpuMat*>(ptt);
  bp::list dims;
  dims.append(dev_src_image_->rows);
  dims.append(dev_src_image_->cols);
  return dims;
}

void PyNet::ReleaseImageData() {
  CUDA_CHECK(cudaFree(dev_mean_image_));
  dev_src_image_->release();
  //dev_stt_image_->release();
  delete dev_src_image_;
  //delete dev_stt_image_;
}

void PyNet::ForwardRegions(bp::list boxes, int context_pad) {
  ForwardRegionsOnSource(boxes, context_pad, 0);
}

void PyNet::ForwardRegionsOnSource(bp::list boxes, int context_pad, int src /*, const string& imageName*/){
  int totalBoxes = len(boxes);
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  // Prepare boxes coordinates
  int ** data;
  data = new int*[totalBoxes];
  for(int j = 0; j < totalBoxes; ++j) {
    data[j] = new int[4];
    bp::list box(boxes[j]);
    for(int k = 0; k < len(box); ++k) {
      data[j][k] = boost::python::extract<int>(box[k]);
    }
  }
  // Crop and Resize boxes in the GPU
  float* dev_blob;
  if (src == 0) {
    dev_blob = CropAndResizeBoxes_GpuMat<float>(dev_src_image_, data, totalBoxes, context_pad, cropsize_, dev_mean_image_);
  } else {
    //dev_blob = CropAndResizeBoxes_GpuMat<float>(dev_stt_image_, data, totalBoxes, context_pad, cropsize_, dev_mean_image_);
  }
  //float* dev_blob = CropAndResizeBoxes_Debug<float>(imageName, data, totalBoxes, context_pad, cropsize_, dev_mean_image_);

  // Copy data to the network
  for (int j = 0; j < input_blobs.size(); ++j) {
    cudaMemcpy(input_blobs[j]->mutable_gpu_data(), dev_blob,
        sizeof(float) * input_blobs[j]->count(), cudaMemcpyDeviceToDevice);
  }
  // Forward the network
  //const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  net_->ForwardPrefilled();

  // Free temporary memory
  for(int j = 0; j < totalBoxes; ++j) delete data[j];
  delete data;
  CUDA_CHECK(cudaFree(dev_blob)); 
}

void PyNet::ForwardRegionsAndState(bp::list boxes, int context_pad, bp::list state /*, const string& imageName*/){
  int totalBoxes = len(boxes);
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  // Prepare boxes coordinates
  int ** data;
  data = new int*[totalBoxes];
  for(int j = 0; j < totalBoxes; ++j) {
    data[j] = new int[4];
    bp::list box(boxes[j]);
    for(int k = 0; k < len(box); ++k) {
      data[j][k] = boost::python::extract<int>(box[k]);
    }
  }
  // Crop and Resize boxes in the GPU
  float* dev_blob = CropAndResizeBoxes_GpuMat<float>(dev_src_image_, data, totalBoxes, context_pad, cropsize_, dev_mean_image_);
  //float* dev_blob = CropAndResizeBoxes_Debug<float>(imageName, data, totalBoxes, context_pad, cropsize_, dev_mean_image_);

  // Copy data to the network
  // Copy Image Data (blobs[0])
    cudaMemcpy(input_blobs[0]->mutable_gpu_data(), dev_blob,
        sizeof(float) * input_blobs[0]->count(), cudaMemcpyDeviceToDevice);
  // Copy State Representation (blobs[1])
  bp::object elem = state[0];
  PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
  //check_array_against_blob(arr, input_blobs[1]);
  cudaMemcpy(input_blobs[1]->mutable_gpu_data(), PyArray_DATA(arr),
      sizeof(float) * input_blobs[1]->count(), cudaMemcpyHostToDevice);

  // Forward the network
  //const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  net_->ForwardPrefilled();

  // Free temporary memory
  for(int j = 0; j < totalBoxes; ++j) delete data[j];
  delete data;
  CUDA_CHECK(cudaFree(dev_blob)); 
}

void PyNet::CoverRegions(bp::list boxes, const string& otherImage, int nameid){
  int totalBoxes = len(boxes);
  // Prepare boxes coordinates
  int ** data;
  data = new int*[totalBoxes];
  for(int j = 0; j < totalBoxes; ++j) {
    data[j] = new int[4];
    bp::list box(boxes[j]);
    for(int k = 0; k < len(box); ++k) {
      data[j][k] = boost::python::extract<int>(box[k]);
    }
  }
  // Cover boxes in the GPU image
  string empty("");
  if (empty.compare(otherImage) != 0) {
    void* prt = LoadImageToGpuMat(otherImage);
    cv::gpu::GpuMat* dev_other = static_cast<cv::gpu::GpuMat*>(prt);
    CoverBoxes_GpuMat(dev_src_image_, dev_other, data, totalBoxes, nameid, false);
    dev_other->release();
    delete dev_other;
  } else {
    CoverBoxes_GpuMat(dev_src_image_, NULL, data, totalBoxes, nameid, true);
  }

  // Free temporary memory
  for(int j = 0; j < totalBoxes; ++j) delete data[j];
  delete data;
}


void PyNet::set_input_arrays(bp::object data_obj, bp::object labels_obj) {
  // check that this network has an input QMemoryDataLayer
  shared_ptr<QMemoryDataLayer<float> > md_layer =
    boost::dynamic_pointer_cast<QMemoryDataLayer<float> >(net_->layers()[0]);
  if (!md_layer) {
    throw std::runtime_error("set_input_arrays may only be called if the"
        " first layer is a QMemoryDataLayer");
  }

  // check that we were passed appropriately-sized contiguous memory
  PyArrayObject* data_arr =
      reinterpret_cast<PyArrayObject*>(data_obj.ptr());
  PyArrayObject* labels_arr =
      reinterpret_cast<PyArrayObject*>(labels_obj.ptr());
  check_contiguous_array(data_arr, "data array", md_layer->datum_channels(),
      md_layer->datum_height(), md_layer->datum_width());
  check_contiguous_array(labels_arr, "labels array", 3, 1, 1);
  if (PyArray_DIMS(data_arr)[0] != PyArray_DIMS(labels_arr)[0]) {
    throw std::runtime_error("data and labels must have the same first"
        " dimension");
  }
  if (PyArray_DIMS(data_arr)[0] % md_layer->batch_size() != 0) {
    throw std::runtime_error("first dimensions of input arrays must be a"
        " multiple of batch size");
  }

  // hold references
  input_data_ = data_obj;
  input_labels_ = labels_obj;
  md_layer->Reset(static_cast<float*>(PyArray_DATA(data_arr)),
      static_cast<float*>(PyArray_DATA(labels_arr)),
      PyArray_DIMS(data_arr)[0]);
}

PySGDSolver::PySGDSolver(const string& param_file) {
  // as in PyNet, (as a convenience, not a guarantee), create a Python
  // exception if param_file can't be opened
  CheckFile(param_file);
  solver_.reset(new SGDSolver<float>(param_file));
  // we need to explicitly store the net wrapper, rather than constructing
  // it on the fly, so that it can hold references to Python objects
  net_.reset(new PyNet(solver_->net()));
}

void PySGDSolver::SolveResume(const string& resume_file) {
  CheckFile(resume_file);
  return solver_->Solve(resume_file);
}

BOOST_PYTHON_MODULE(_caffe) {
  // below, we prepend an underscore to methods that will be replaced
  // in Python
  bp::class_<PyNet, shared_ptr<PyNet> >(
      "Net", bp::init<string, string>())
      .def(bp::init<string>())
      .def("_forward",              &PyNet::Forward)
      .def("_backward",             &PyNet::Backward)
      .def("reshape",               &PyNet::Reshape)
      .def("set_mode_cpu",          &PyNet::set_mode_cpu)
      .def("set_mode_gpu",          &PyNet::set_mode_gpu)
      .def("set_phase_train",       &PyNet::set_phase_train)
      .def("set_phase_test",        &PyNet::set_phase_test)
      .def("set_device",            &PyNet::set_device)
      .def("ForwardRegions",        &PyNet::ForwardRegions)
      .def("ForwardRegionsOnSource",&PyNet::ForwardRegionsOnSource)
      .def("ForwardRegionsAndState",&PyNet::ForwardRegionsAndState)
      .def("InitializeImage",       &PyNet::InitializeImage)
      .def("ReleaseImageData",      &PyNet::ReleaseImageData)
      .def("CoverRegions",          &PyNet::CoverRegions)
      .add_property("_blobs",       &PyNet::blobs)
      .add_property("layers",       &PyNet::layers)
      .add_property("_blob_names",  &PyNet::blob_names)
      .add_property("_layer_names", &PyNet::layer_names)
      .add_property("inputs",       &PyNet::inputs)
      .add_property("outputs",      &PyNet::outputs)
      .add_property("mean",         &PyNet::mean_)
      .add_property("input_scale",  &PyNet::input_scale_)
      .add_property("raw_scale",    &PyNet::raw_scale_)
      .add_property("channel_swap", &PyNet::channel_swap_)
      .def("_set_input_arrays",     &PyNet::set_input_arrays)
      .def("save",                  &PyNet::save);

  bp::class_<PyBlob<float>, PyBlobWrap>(
      "Blob", bp::no_init)
      .add_property("num",      &PyBlob<float>::num)
      .add_property("channels", &PyBlob<float>::channels)
      .add_property("height",   &PyBlob<float>::height)
      .add_property("width",    &PyBlob<float>::width)
      .add_property("count",    &PyBlob<float>::count)
      .def("reshape",           &PyBlob<float>::Reshape)
      .add_property("data",     &PyBlobWrap::get_data)
      .add_property("diff",     &PyBlobWrap::get_diff);

  bp::class_<PyLayer>(
      "Layer", bp::no_init)
      .add_property("blobs", &PyLayer::blobs);

  bp::class_<PySGDSolver, boost::noncopyable>(
      "SGDSolver", bp::init<string>())
      .add_property("net", &PySGDSolver::net)
      .add_property("test_nets", &PySGDSolver::test_nets)
      .add_property("iter",      &PySGDSolver::iter)
      .def("change_lr",    &PySGDSolver::ChangeLearningRage)
      .def("solve",        &PySGDSolver::Solve)
      .def("solve",        &PySGDSolver::SolveResume);

  bp::class_<vector<shared_ptr<PyNet> > >("NetVec")
      .def(bp::vector_indexing_suite<vector<shared_ptr<PyNet> >, true>());

  bp::class_<vector<PyBlob<float> > >("BlobVec")
      .def(bp::vector_indexing_suite<vector<PyBlob<float> >, true>());

  bp::class_<vector<PyLayer> >("LayerVec")
      .def(bp::vector_indexing_suite<vector<PyLayer>, true>());

  bp::class_<vector<string> >("StringVec")
      .def(bp::vector_indexing_suite<vector<string> >());
  import_array();
}

}  // namespace caffe

