Pod::Spec.new do |s|
  s.name             = "Bender"
  s.version          = "0.1.0"
  s.summary          = "Bender is an abstraction layer over MetalPerformanceShaders which is used to work with neural networks."
  s.homepage         = "https://github.com/xmartlabs/Bender"
  s.license          = { type: 'MIT', file: 'LICENSE' }
  s.author           = { "Xmartlabs SRL" => "swift@xmartlabs.com" }
  s.source           = { git: "https://github.com/xmartlabs/Bender.git", tag: s.version.to_s }
  s.social_media_url = 'https://twitter.com/xmartlabs'
  s.ios.deployment_target = '10.0'
  s.requires_arc = true
  s.ios.source_files = 'Sources/**/*.{swift,metal}'
  s.dependency 'SwiftProtobuf', '~> 0.9.903'
end
