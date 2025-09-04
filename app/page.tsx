'use client';

import { useState, useRef, useCallback } from 'react';

const VIDEO_EFFECTS = [
  { id: 'blob_detect', name: 'Blob Detection', description: 'Detect and highlight blob-like objects' },
  { id: 'cryptomatte_video', name: 'Cryptomatte', description: 'Advanced object masking' },
  { id: 'dotpad', name: 'Dot Pattern', description: 'Apply dot pattern overlay' },
  { id: 'object_connectors_video', name: 'Object Connectors', description: 'Connect detected objects with lines' },
  { id: 'object_specific_blobs', name: 'Specific Blobs', description: 'Target specific blob objects' },
  { id: 'ruttra_video', name: 'Ruttra Effect', description: 'Apply Ruttra visual effect' },
  { id: 'splinterframes', name: 'Splinter Frames', description: 'Create frame splinter effect' },
  // New multi-effects from effects_video.py
  { id: 'trail', name: 'Motion Trails', description: 'Create fading trails behind moving objects' },
  { id: 'echo', name: 'Echo Frames', description: 'Blend multiple frames for ghosting effect' },
  { id: 'flow', name: 'Flow Fields', description: 'Optical flow warping and distortion' },
  { id: 'ascii', name: 'ASCII Art', description: 'Convert video to ASCII character overlay' },
  { id: 'particles', name: 'Particle Trails', description: 'Spawn particles from moving objects' },
  { id: 'bbox_mesh', name: 'Mesh Overlay', description: 'Draw mesh grids over detected objects' },
];

// Comprehensive parameter configuration for all effects
const EFFECT_PARAMETERS = {
  blob_detect: [
    { name: 'canny_sigma', type: 'range', min: 0.1, max: 1.0, step: 0.01, default: 0.33, description: 'Edge sensitivity - lower values detect more edges' },
    { name: 'blur_ksize', type: 'range', min: 1, max: 15, step: 2, default: 3, description: 'Blur kernel size - reduces noise but softens edges' },
    { name: 'dilate_ksize', type: 'range', min: 1, max: 15, step: 2, default: 3, description: 'Edge thickening - makes detected edges thicker' },
    { name: 'min_contour_area', type: 'range', min: 5, max: 200, step: 5, default: 20, description: 'Minimum blob size - filters out tiny specks' },
    { name: 'border_margin', type: 'range', min: 0, max: 50, step: 1, default: 6, description: 'Border exclusion - ignore blobs near edges' },
    { name: 'iou_threshold', type: 'range', min: 0.05, max: 0.8, step: 0.05, default: 0.20, description: 'Box merging threshold - higher merges more overlapping boxes' },
  ],
  cryptomatte_video: [
    { name: 'k_clusters', type: 'range', min: 2, max: 16, step: 1, default: 6, description: 'Number of color regions - more clusters = more detailed segmentation' },
    { name: 'blur_ksize', type: 'range', min: 0, max: 15, step: 1, default: 3, description: 'Pre-processing blur - smooths colors before clustering' },
    { name: 'max_iters', type: 'range', min: 1, max: 20, step: 1, default: 5, description: 'Clustering iterations - more iterations = better accuracy but slower' },
    { name: 'downscale', type: 'range', min: 0.25, max: 1.0, step: 0.05, default: 1.0, description: 'Processing scale - lower values process faster but less detailed' },
    { name: 'colorspace', type: 'select', options: ['lab', 'hsv', 'rgb'], default: 'lab', description: 'Color space for clustering - LAB usually works best' },
  ],
  dotpad: [
    { name: 'cell_px', type: 'range', min: 4, max: 32, step: 2, default: 12, description: 'Dot grid size - larger cells = bigger dots, faster processing' },
    { name: 'dot_scale', type: 'range', min: 0.1, max: 1.0, step: 0.05, default: 0.95, description: 'Maximum dot size relative to cell' },
    { name: 'gamma', type: 'range', min: 0.2, max: 3.0, step: 0.1, default: 1.0, description: 'Brightness response - >1 emphasizes darks, <1 emphasizes brights' },
    { name: 'color_mode', type: 'select', options: ['color', 'mono'], default: 'color', description: 'Color or monochrome dots' },
    { name: 'invert_bright', type: 'checkbox', default: false, description: 'Invert brightness mapping - bright areas become small dots' },
    { name: 'draw_grid', type: 'checkbox', default: false, description: 'Show grid lines over dots' },
  ],
  object_connectors_video: [
    { name: 'conf_thresh', type: 'range', min: 0.1, max: 0.9, step: 0.05, default: 0.35, description: 'Object detection confidence - higher = fewer but more certain detections' },
    { name: 'k_neighbors', type: 'range', min: 1, max: 8, step: 1, default: 2, description: 'Connections per object - how many lines each object gets' },
    { name: 'max_connect_dist', type: 'range', min: 0.1, max: 1.0, step: 0.05, default: 0.45, description: 'Maximum connection distance - limits how far objects can connect' },
    { name: 'curve_bulge', type: 'range', min: 0.0, max: 0.5, step: 0.02, default: 0.18, description: 'Line curviness - 0 = straight lines, higher = more curved' },
    { name: 'dot_spacing', type: 'range', min: 3, max: 30, step: 1, default: 10, description: 'Space between dots in connection lines' },
    { name: 'dot_radius', type: 'range', min: 1, max: 8, step: 1, default: 2, description: 'Size of connection dots' },
  ],
  object_specific_blobs: [
    { name: 'confidence_threshold', type: 'range', min: 0.1, max: 0.9, step: 0.05, default: 0.3, description: 'Object detection confidence threshold' },
    { name: 'bright_threshold', type: 'range', min: 100, max: 255, step: 5, default: 180, description: 'Brightness threshold for detection' },
    { name: 'saturation_threshold', type: 'range', min: 20, max: 255, step: 5, default: 80, description: 'Color saturation threshold for colorful objects' },
    { name: 'motion_threshold', type: 'range', min: 10, max: 100, step: 5, default: 30, description: 'Motion sensitivity for moving objects' },
    { name: 'min_contour_area', type: 'range', min: 5, max: 100, step: 5, default: 20, description: 'Minimum blob size to detect' },
  ],
  ruttra_video: [
    { name: 'step_x', type: 'range', min: 1, max: 16, step: 1, default: 4, description: 'Horizontal sampling step - lower = more detailed but slower' },
    { name: 'step_y', type: 'range', min: 1, max: 16, step: 1, default: 4, description: 'Vertical sampling step - lower = more detailed but slower' },
    { name: 'height_gain', type: 'range', min: 0.1, max: 2.0, step: 0.05, default: 0.35, description: 'Height displacement amount - how much brightness affects line height' },
  ],
  splinterframes: [
    { name: 'canny_sigma', type: 'range', min: 0.1, max: 1.0, step: 0.01, default: 0.33, description: 'Edge detection sensitivity' },
    { name: 'rgb_shift_max', type: 'range', min: 0, max: 20, step: 1, default: 5, description: 'Color channel shift amount for glitch effect' },
    { name: 'displace_strength', type: 'range', min: 0, max: 30, step: 1, default: 8, description: 'Noise warp strength inside detected regions' },
    { name: 'min_contour_area', type: 'range', min: 5, max: 200, step: 5, default: 20, description: 'Minimum area for effect regions' },
    { name: 'invert_only', type: 'checkbox', default: true, description: 'Only invert colors (disable glitch effects)' },
    { name: 'include_glitch', type: 'checkbox', default: false, description: 'Add RGB shift and displacement glitch effects' },
  ],
  // Multi-effects from effects_video.py
  trail: [
    { name: 'decay', type: 'range', min: 0.5, max: 0.99, step: 0.01, default: 0.92, description: 'Trail fade speed - higher = longer lasting trails' },
  ],
  echo: [
    { name: 'history', type: 'range', min: 2, max: 15, step: 1, default: 6, description: 'Number of frames to blend - more frames = longer echo effect' },
  ],
  flow: [
    { name: 'flow_strength', type: 'range', min: 0.1, max: 2.0, step: 0.1, default: 0.8, description: 'Flow field strength - higher = more distortion' },
    { name: 'show_vectors', type: 'checkbox', default: false, description: 'Show flow direction vectors as white lines' },
  ],
  ascii: [
    { name: 'ascii_cell', type: 'range', min: 4, max: 24, step: 2, default: 8, description: 'Character cell size - smaller = more detailed ASCII' },
    { name: 'alpha', type: 'range', min: 0.1, max: 1.0, step: 0.05, default: 0.85, description: 'ASCII overlay opacity - lower = more transparent' },
  ],
  particles: [
    { name: 'spawn', type: 'range', min: 5, max: 100, step: 5, default: 25, description: 'Particles spawned per moving object - more = denser trails' },
  ],
  bbox_mesh: [
    { name: 'grid_x', type: 'range', min: 2, max: 20, step: 1, default: 8, description: 'Horizontal mesh divisions' },
    { name: 'grid_y', type: 'range', min: 2, max: 20, step: 1, default: 8, description: 'Vertical mesh divisions' },
    { name: 'alpha', type: 'range', min: 0.1, max: 1.0, step: 0.05, default: 0.85, description: 'Mesh overlay opacity' },
  ],
};

type ProcessingState = 'idle' | 'processing' | 'completed' | 'error';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedEffect, setSelectedEffect] = useState<string>('');
  const [objectType, setObjectType] = useState<string>('');
  const [effectParameters, setEffectParameters] = useState<{[key: string]: any}>({});
  const [processingState, setProcessingState] = useState<ProcessingState>('idle');
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize parameters when effect changes
  const initializeParameters = (effectId: string) => {
    if (effectId && EFFECT_PARAMETERS[effectId as keyof typeof EFFECT_PARAMETERS]) {
      const params = EFFECT_PARAMETERS[effectId as keyof typeof EFFECT_PARAMETERS];
      const initialParams: {[key: string]: any} = {};
      params.forEach(param => {
        initialParams[param.name] = param.default;
      });
      setEffectParameters(initialParams);
    } else {
      setEffectParameters({});
    }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      const file = files[0];
      if (file.type === 'video/mp4' || file.type === 'video/quicktime' || file.name.endsWith('.mov') || file.name.endsWith('.mp4')) {
        setSelectedFile(file);
        setProcessedVideoUrl(null);
        setProcessingState('idle');
      } else {
        alert('Please select a .mov or .mp4 video file');
      }
    }
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      const file = files[0];
      if (file.type === 'video/mp4' || file.type === 'video/quicktime' || file.name.endsWith('.mov') || file.name.endsWith('.mp4')) {
        setSelectedFile(file);
        setProcessedVideoUrl(null);
        setProcessingState('idle');
      } else {
        alert('Please select a .mov or .mp4 video file');
      }
    }
  };

  const processVideo = async () => {
    if (!selectedFile || !selectedEffect) {
      alert('Please select a video file and an effect');
      return;
    }

    // Check if object type is required for specific blobs effect
    if (selectedEffect === 'object_specific_blobs' && !objectType.trim()) {
      alert('Please specify the object type for the Specific Blobs effect');
      return;
    }

    setProcessingState('processing');
    
    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('effect', selectedEffect);
    if (selectedEffect === 'object_specific_blobs') {
      formData.append('objectType', objectType.trim());
    }
    
    // Add effect parameters
    formData.append('parameters', JSON.stringify(effectParameters));

    try {
      const response = await fetch('/api/process-video', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setProcessedVideoUrl(url);
        setProcessingState('completed');
      } else {
        throw new Error('Processing failed');
      }
    } catch (error) {
      console.error('Error processing video:', error);
      setProcessingState('error');
      alert('Error processing video. Please try again.');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && selectedFile && selectedEffect && processingState === 'idle') {
      // For specific blobs effect, also check if object type is provided
      if (selectedEffect === 'object_specific_blobs' && !objectType.trim()) {
        return;
      }
      processVideo();
    }
  };

  const downloadVideo = () => {
    if (processedVideoUrl) {
      const a = document.createElement('a');
      a.href = processedVideoUrl;
      a.download = `processed_${selectedFile?.name || 'video'}.mp4`;
      a.click();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-8">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Video Effects Studio
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Upload your video, choose an effect, and create something amazing
          </p>
        </header>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8 space-y-8">
          {/* File Upload Area */}
          <div className="space-y-4">
            <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
              Upload Video
            </h2>
            <div
              className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200 ${
                dragActive
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : selectedFile
                  ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                  : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mov,.mp4,video/mp4,video/quicktime"
                onChange={handleFileInput}
                className="hidden"
              />
              
              {selectedFile ? (
                <div className="space-y-2">
                  <div className="text-6xl">✅</div>
                  <p className="text-lg font-medium text-green-700 dark:text-green-300">
                    {selectedFile.name}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="text-6xl text-gray-400">📁</div>
                  <div>
                    <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
                      Drop your video here or{' '}
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="text-blue-600 dark:text-blue-400 hover:underline"
                      >
                        browse files
                      </button>
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                      Supports .mov and .mp4 files
                    </p>
                  </div>
                </div>
              )}
            </div>
        </div>

          {/* Effects Dropdown */}
          <div className="space-y-4">
            <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
              Choose Effect
            </h2>
            <select
              value={selectedEffect}
              onChange={(e) => {
                const newEffect = e.target.value;
                setSelectedEffect(newEffect);
                // Reset object type when changing effects
                if (newEffect !== 'object_specific_blobs') {
                  setObjectType('');
                }
                // Initialize parameters for the new effect
                initializeParameters(newEffect);
              }}
              onKeyPress={handleKeyPress}
              className="w-full p-4 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
            >
              <option value="">Select an effect...</option>
              {VIDEO_EFFECTS.map((effect) => (
                <option key={effect.id} value={effect.id}>
                  {effect.name} - {effect.description}
                </option>
              ))}
            </select>
          </div>

          {/* Conditional Object Type Input */}
          {selectedEffect === 'object_specific_blobs' && (
            <div className="space-y-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6">
              <div className="flex items-center space-x-2">
                <div className="text-2xl">🎯</div>
                <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-100">
                  Specify Target Object
                </h3>
              </div>
              <p className="text-blue-700 dark:text-blue-300 text-sm">
                Enter the type of object you want to detect and track (e.g., "snow", "fireworks", "person", "car", etc.)
              </p>
              <input
                type="text"
                value={objectType}
                onChange={(e) => setObjectType(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="e.g., snow, fireworks, person, car..."
                className="w-full p-4 border border-blue-300 dark:border-blue-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg placeholder-gray-500 dark:placeholder-gray-400"
              />
              <div className="text-xs text-blue-600 dark:text-blue-400">
                <strong>Popular objects:</strong> person, car, bicycle, dog, cat, bird, airplane, boat, traffic light, fire hydrant, stop sign, bench, umbrella, handbag, bottle, chair, dining table, tv, laptop, cell phone, sports ball, kite, frisbee, snowboard, skis
              </div>
            </div>
          )}

          {/* Effect Parameters */}
          {selectedEffect && EFFECT_PARAMETERS[selectedEffect as keyof typeof EFFECT_PARAMETERS] && (
            <div className="space-y-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-xl p-6">
              <div className="flex items-center space-x-2">
                <div className="text-2xl">⚙️</div>
                <h3 className="text-xl font-semibold text-purple-900 dark:text-purple-100">
                  Effect Parameters
                </h3>
              </div>
              <p className="text-purple-700 dark:text-purple-300 text-sm">
                Fine-tune your effect settings for the perfect result
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {EFFECT_PARAMETERS[selectedEffect as keyof typeof EFFECT_PARAMETERS].map((param) => (
                  <div key={param.name} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium text-purple-900 dark:text-purple-100 capitalize">
                        {param.name.replace(/_/g, ' ')}
                      </label>
                      {param.type === 'range' && (
                        <span className="text-xs text-purple-600 dark:text-purple-400 font-mono">
                          {effectParameters[param.name] || param.default}
                        </span>
                      )}
                    </div>
                    
                    {param.type === 'range' && (
                      <div className="space-y-1">
                        <input
                          type="range"
                          min={param.min}
                          max={param.max}
                          step={param.step}
                          value={effectParameters[param.name] || param.default}
                          onChange={(e) => setEffectParameters(prev => ({
                            ...prev,
                            [param.name]: parseFloat(e.target.value)
                          }))}
                          className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer dark:bg-purple-700 slider"
                        />
                        <div className="flex justify-between text-xs text-purple-600 dark:text-purple-400">
                          <span>{param.min}</span>
                          <span>{param.max}</span>
                        </div>
                      </div>
                    )}
                    
                    {param.type === 'select' && (
                      <select
                        value={effectParameters[param.name] || param.default}
                        onChange={(e) => setEffectParameters(prev => ({
                          ...prev,
                          [param.name]: e.target.value
                        }))}
                        className="w-full p-2 border border-purple-300 dark:border-purple-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
                      >
                        {param.options?.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    )}
                    
                    {param.type === 'checkbox' && (
                      <label className="flex items-center space-x-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={effectParameters[param.name] ?? param.default}
                          onChange={(e) => setEffectParameters(prev => ({
                            ...prev,
                            [param.name]: e.target.checked
                          }))}
                          className="w-4 h-4 text-purple-600 bg-gray-100 border-gray-300 rounded focus:ring-purple-500 dark:focus:ring-purple-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600"
                        />
                        <span className="text-sm text-purple-700 dark:text-purple-300">
                          {param.description}
                        </span>
                      </label>
                    )}
                    
                    {param.type !== 'checkbox' && (
                      <p className="text-xs text-purple-600 dark:text-purple-400 leading-relaxed">
                        {param.description}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Process Button */}
          <div className="flex justify-center">
            <button
              onClick={processVideo}
              disabled={
                !selectedFile || 
                !selectedEffect || 
                processingState === 'processing' ||
                (selectedEffect === 'object_specific_blobs' && !objectType.trim())
              }
              className="px-8 py-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold rounded-xl transition-colors duration-200 text-lg min-w-[200px]"
            >
              {processingState === 'processing' ? (
                <div className="flex items-center justify-center space-x-2">
                  <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                  <span>Processing...</span>
                </div>
              ) : (
                'Apply Effect'
              )}
            </button>
          </div>

          {/* Processing Status */}
          {processingState === 'processing' && (
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-6 text-center">
              <div className="animate-pulse">
                <div className="text-2xl mb-2">🎬</div>
                <p className="text-blue-700 dark:text-blue-300 font-medium">
                  Your video is being processed...
                </p>
                <p className="text-sm text-blue-600 dark:text-blue-400 mt-1">
                  This may take a few minutes depending on video length
                </p>
              </div>
            </div>
          )}

          {/* Download Button */}
          {processingState === 'completed' && processedVideoUrl && (
            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl p-6 text-center space-y-4">
              <div className="text-2xl">🎉</div>
              <p className="text-green-700 dark:text-green-300 font-medium text-lg">
                Your video is ready!
              </p>
              <button
                onClick={downloadVideo}
                className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-xl transition-colors duration-200"
              >
                Download Processed Video
              </button>
            </div>
          )}

          {/* Error State */}
          {processingState === 'error' && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6 text-center">
              <div className="text-2xl mb-2">❌</div>
              <p className="text-red-700 dark:text-red-300 font-medium">
                Something went wrong while processing your video
              </p>
              <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                Please try again or contact support
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
