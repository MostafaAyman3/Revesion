import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

const modalityColors = {
  flair: 'from-blue-500 to-blue-600',
  t1: 'from-green-500 to-green-600',
  t1ce: 'from-purple-500 to-purple-600',
  t2: 'from-orange-500 to-orange-600'
}

const modalityBorders = {
  flair: 'border-blue-500/50 hover:border-blue-400',
  t1: 'border-green-500/50 hover:border-green-400',
  t1ce: 'border-purple-500/50 hover:border-purple-400',
  t2: 'border-orange-500/50 hover:border-orange-400'
}

export default function FileUpload({ modality, label, file, onFileChange, description }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const uploadedFile = acceptedFiles[0]
      const fileName = uploadedFile.name.toLowerCase()
      
      // Validate file extension
      if (fileName.endsWith('.nii') || fileName.endsWith('.nii.gz')) {
        onFileChange(modality, uploadedFile)
      } else {
        alert('Please upload a .nii or .nii.gz file')
      }
    }
  }, [modality, onFileChange])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/gzip': ['.nii.gz'],
      'application/octet-stream': ['.nii']
    },
    maxFiles: 1
  })

  const removeFile = (e) => {
    e.stopPropagation()
    onFileChange(modality, null)
  }

  return (
    <div
      {...getRootProps()}
      className={`relative p-4 rounded-xl border-2 border-dashed transition-all duration-300 cursor-pointer
        ${file 
          ? 'bg-gray-800/50 border-gray-600' 
          : `bg-gray-900/50 ${modalityBorders[modality]}`}
        ${isDragActive ? 'scale-105 bg-gray-800/70' : ''}`}
    >
      <input {...getInputProps()} />
      
      {/* Modality Badge */}
      <div className={`absolute -top-3 left-4 px-3 py-1 rounded-full text-xs font-bold text-white bg-gradient-to-r ${modalityColors[modality]}`}>
        {label}
      </div>

      <div className="pt-2">
        {file ? (
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">{file.name}</p>
              <p className="text-xs text-gray-400">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
            <button
              onClick={removeFile}
              className="p-1 hover:bg-red-500/20 rounded transition-colors"
            >
              <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        ) : (
          <div className="text-center py-4">
            <svg className="w-10 h-10 mx-auto mb-2 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-sm text-gray-400 mb-1">
              {isDragActive ? 'Drop file here' : 'Drag & drop or click'}
            </p>
            <p className="text-xs text-gray-500">{description}</p>
            <p className="text-xs text-gray-600 mt-1">.nii / .nii.gz</p>
          </div>
        )}
      </div>
    </div>
  )
}
