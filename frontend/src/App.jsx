import { useState } from 'react'
import Header from './components/Header'
import FileUpload from './components/FileUpload'
import LoadingState from './components/LoadingState'
import Visualization3D from './components/Visualization3D'
import TumorStats from './components/TumorStats'
import { runInference } from './api/inference'

function App() {
  const [files, setFiles] = useState({
    flair: null,
    t1: null,
    t1ce: null,
    t2: null
  })
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (modality, file) => {
    setFiles(prev => ({ ...prev, [modality]: file }))
    setError(null)
  }

  const handleClearFiles = () => {
    setFiles({ flair: null, t1: null, t1ce: null, t2: null })
    setResult(null)
    setError(null)
  }

  const allFilesUploaded = files.flair && files.t1 && files.t1ce && files.t2

  const handleRunInference = async () => {
    if (!allFilesUploaded) {
      setError('Please upload all 4 MRI modalities')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await runInference(files)
      setResult(response)
    } catch (err) {
      setError(err.message || 'Inference failed. Please check your files and try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 container mx-auto px-4 py-8 max-w-7xl">
        {/* Upload Section */}
        <section className="mb-8">
          <div className="glass rounded-2xl p-6 glow-blue">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <svg className="w-7 h-7 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              Upload MRI Modalities
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <FileUpload 
                modality="flair" 
                label="FLAIR"
                file={files.flair}
                onFileChange={handleFileChange}
                description="Fluid-attenuated inversion recovery"
              />
              <FileUpload 
                modality="t1" 
                label="T1"
                file={files.t1}
                onFileChange={handleFileChange}
                description="T1-weighted MRI"
              />
              <FileUpload 
                modality="t1ce" 
                label="T1ce"
                file={files.t1ce}
                onFileChange={handleFileChange}
                description="T1 contrast-enhanced"
              />
              <FileUpload 
                modality="t2" 
                label="T2"
                file={files.t2}
                onFileChange={handleFileChange}
                description="T2-weighted MRI"
              />
            </div>

            {error && (
              <div className="mb-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
                <span className="font-medium">Error:</span> {error}
              </div>
            )}

            <div className="flex gap-4 justify-center">
              <button
                onClick={handleRunInference}
                disabled={!allFilesUploaded || isLoading}
                className={`px-8 py-3 rounded-xl font-semibold text-lg transition-all duration-300 flex items-center gap-2
                  ${allFilesUploaded && !isLoading 
                    ? 'bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white shadow-lg hover:shadow-blue-500/25' 
                    : 'bg-gray-600 text-gray-400 cursor-not-allowed'}`}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
                Run Segmentation
              </button>
              
              <button
                onClick={handleClearFiles}
                className="px-6 py-3 rounded-xl font-medium border border-gray-600 text-gray-300 hover:bg-gray-700/50 transition-all"
              >
                Clear All
              </button>
            </div>
          </div>
        </section>

        {/* Loading State */}
        {isLoading && <LoadingState />}

        {/* Results Section */}
        {result && !isLoading && (
          <section className="space-y-6">
            {/* Statistics */}
            <TumorStats 
              tumorVolumes={result.tumor_volumes} 
              volumeShape={result.volume_shape}
            />

            {/* 3D Visualization */}
            <div className="glass rounded-2xl p-6">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
                <svg className="w-7 h-7 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
                </svg>
                3D Tumor Visualization
              </h2>
              <Visualization3D data={result.visualization_data} />
            </div>

            {/* Legend */}
            <div className="glass rounded-xl p-4">
              <div className="flex flex-wrap justify-center gap-6">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-red-500 glow-red"></div>
                  <span className="text-sm">Necrotic Core (Class 1)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-green-500 glow-green"></div>
                  <span className="text-sm">Edema (Class 2)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-yellow-500 glow-gold"></div>
                  <span className="text-sm">Enhancing Tumor (Class 4)</span>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="py-4 text-center text-gray-500 text-sm">
        <p>INSTANT-ODC AI Hackathon 2026 | Brain Tumor Segmentation</p>
      </footer>
    </div>
  )
}

export default App
