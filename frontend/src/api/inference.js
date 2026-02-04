import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function runInference(files) {
  const formData = new FormData()
  
  formData.append('flair', files.flair)
  formData.append('t1', files.t1)
  formData.append('t1ce', files.t1ce)
  formData.append('t2', files.t2)
  formData.append('return_rle', 'true')

  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5 minutes timeout for large files
    })

    if (!response.data.success) {
      throw new Error(response.data.message || 'Inference failed')
    }

    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.response) {
        throw new Error(error.response.data?.detail || `Server error: ${error.response.status}`)
      } else if (error.request) {
        throw new Error('Cannot connect to server. Please ensure the backend is running.')
      }
    }
    throw error
  }
}

export async function checkHealth() {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`)
    return response.data
  } catch (error) {
    return { status: 'unhealthy', model_loaded: false, device: 'unknown' }
  }
}
