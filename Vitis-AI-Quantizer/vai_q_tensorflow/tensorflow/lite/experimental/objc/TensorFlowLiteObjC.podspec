Pod::Spec.new do |s|
  s.name             = 'TensorFlowLiteObjC'
  s.version          = '1.14.0'
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.source           = { :git => 'https://github.com/tensorflow/tensorflow.git', :tag => "v#{s.version}" }
  s.summary          = 'TensorFlow Lite for Objective-C'
  s.description      = <<-DESC

  TensorFlow Lite is TensorFlow's lightweight solution for Objective-C
  developers. It enables low-latency inference of on-device machine learning
  models with a small binary size and fast performance supporting hardware
  acceleration.
                       DESC

  s.ios.deployment_target = '9.0'

  s.module_name = 'TFLTensorFlowLite'
  s.static_framework = true

  tfl_dir = 'tensorflow/lite/'
  objc_dir = tfl_dir + 'experimental/objc/'
  s.public_header_files = objc_dir + 'apis/*.h'
  s.source_files = [
    objc_dir + '{apis,sources}/*.{h,m,mm}',
    tfl_dir + 'experimental/c/c_api.h',
    tfl_dir + 'experimental/c/c_api_types.h',
  ]
  s.module_map = objc_dir + 'apis/framework.modulemap'
  s.dependency 'TensorFlowLiteC', "#{s.version}"
  s.pod_target_xcconfig = {
    'HEADER_SEARCH_PATHS' =>
      '"${PODS_TARGET_SRCROOT}" ' +
      '"${PODS_TARGET_SRCROOT}/' + objc_dir  + 'apis"',
    'VALID_ARCHS' => 'x86_64 armv7 arm64',
  }

  s.test_spec 'Tests' do |ts|
    ts.source_files = objc_dir + 'tests/*.m'
    ts.resources = [
      tfl_dir + 'testdata/add.bin',
      tfl_dir + 'testdata/add_quantized.bin',
    ]
  end
end
