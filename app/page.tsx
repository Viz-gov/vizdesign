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
];

type ProcessingState = 'idle' | 'processing' | 'completed' | 'error';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedEffect, setSelectedEffect] = useState<string>('');
  const [objectType, setObjectType] = useState<string>('');
  const [processingState, setProcessingState] = useState<ProcessingState>('idle');
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
                setSelectedEffect(e.target.value);
                // Reset object type when changing effects
                if (e.target.value !== 'object_specific_blobs') {
                  setObjectType('');
                }
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
