export default function LoadingState() {
  return (
    <div className="glass rounded-2xl p-12 text-center">
      <div className="relative w-24 h-24 mx-auto mb-6">
        {/* Outer ring */}
        <div className="absolute inset-0 border-4 border-blue-500/30 rounded-full"></div>
        
        {/* Spinning ring */}
        <div className="absolute inset-0 border-4 border-transparent border-t-blue-500 rounded-full animate-spin"></div>
        
        {/* Inner brain icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <svg className="w-10 h-10 text-blue-400 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        </div>
      </div>
      
      <h3 className="text-xl font-semibold mb-2">Processing MRI Scans</h3>
      <p className="text-gray-400 mb-6">Running AI-powered tumor segmentation...</p>
      
      <div className="space-y-3 max-w-md mx-auto">
        <div className="flex items-center gap-3 text-sm">
          <div className="w-2 h-2 rounded-full bg-green-500"></div>
          <span className="text-gray-300">Loading NIfTI volumes</span>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <div className="w-2 h-2 rounded-full bg-green-500"></div>
          <span className="text-gray-300">Applying BraTS normalization</span>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
          <span className="text-gray-300">Running slice-wise inference</span>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <div className="w-2 h-2 rounded-full bg-gray-600"></div>
          <span className="text-gray-500">Generating 3D visualization</span>
        </div>
      </div>
    </div>
  )
}
