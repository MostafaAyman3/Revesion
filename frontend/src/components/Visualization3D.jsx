import Plot from 'react-plotly.js'
import { useMemo } from 'react'

export default function Visualization3D({ data }) {
  const plotData = useMemo(() => {
    if (!data) return []

    const traces = []
    
    // Add brain shell mesh (semi-transparent gray)
    if (data.brain_mesh) {
      const brain = data.brain_mesh
      traces.push({
        type: 'mesh3d',
        name: brain.name || 'Brain',
        x: brain.vertices.x,
        y: brain.vertices.y,
        z: brain.vertices.z,
        i: brain.faces.i,
        j: brain.faces.j,
        k: brain.faces.k,
        color: brain.color || 'gray',
        opacity: brain.opacity || 0.1,
        flatshading: true,
        lighting: {
          ambient: 0.8,
          diffuse: 0.5,
          specular: 0.1
        },
        hoverinfo: 'name'
      })
    }

    // Add tumor class meshes
    if (data.meshes && data.meshes.length > 0) {
      data.meshes.forEach(mesh => {
        if (mesh.type === 'scatter') {
          // Fallback to scatter if mesh generation failed
          traces.push({
            type: 'scatter3d',
            mode: 'markers',
            name: mesh.label,
            x: mesh.x,
            y: mesh.y,
            z: mesh.z,
            marker: {
              size: 2,
              color: mesh.color,
              opacity: 0.7
            },
            hovertemplate: `${mesh.label}<br>Voxels: ${mesh.voxel_count?.toLocaleString()}<extra></extra>`
          })
        } else {
          // Proper 3D mesh
          traces.push({
            type: 'mesh3d',
            name: mesh.label,
            x: mesh.vertices.x,
            y: mesh.vertices.y,
            z: mesh.vertices.z,
            i: mesh.faces.i,
            j: mesh.faces.j,
            k: mesh.faces.k,
            color: mesh.color,
            opacity: mesh.opacity || 0.6,
            flatshading: true,
            lighting: {
              ambient: 0.6,
              diffuse: 0.8,
              specular: 0.3,
              roughness: 0.5
            },
            lightposition: {
              x: 100,
              y: 200,
              z: 0
            },
            hoverinfo: 'name',
            showlegend: true
          })
        }
      })
    }
    
    // Legacy support for old "classes" format (scatter)
    if (data.classes && data.classes.length > 0) {
      data.classes.forEach(cls => {
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          name: cls.label,
          x: cls.x,
          y: cls.y,
          z: cls.z,
          marker: {
            size: 2,
            color: cls.color,
            opacity: 0.7
          },
          hovertemplate: `${cls.label}<extra></extra>`
        })
      })
    }

    return traces
  }, [data])

  const layout = useMemo(() => ({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      color: '#e2e8f0'
    },
    scene: {
      xaxis: {
        visible: false,
        showgrid: false,
        zeroline: false
      },
      yaxis: {
        visible: false,
        showgrid: false,
        zeroline: false
      },
      zaxis: {
        visible: false,
        showgrid: false,
        zeroline: false
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.2 }
      },
      aspectmode: 'data',
      bgcolor: 'rgba(26,26,46,0.5)'
    },
    legend: {
      x: 0,
      y: 1,
      bgcolor: 'rgba(26,26,46,0.9)',
      bordercolor: 'rgba(255,255,255,0.2)',
      borderwidth: 1,
      font: { color: '#e2e8f0' }
    },
    margin: { l: 0, r: 0, t: 30, b: 0 },
    autosize: true
  }), [])

  const config = useMemo(() => ({
    displayModeBar: true,
    modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
    displaylogo: false,
    responsive: true
  }), [])

  // Check if there's any valid data
  const hasData = data && (
    (data.meshes && data.meshes.length > 0) || 
    (data.classes && data.classes.length > 0) ||
    data.brain_mesh
  )

  if (!hasData) {
    return (
      <div className="h-[500px] flex items-center justify-center text-gray-400">
        <div className="text-center">
          <svg className="w-16 h-16 mx-auto mb-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M12 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p>No tumor regions detected</p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative">
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '600px' }}
        useResizeHandler={true}
      />
      
      {/* Controls hint */}
      <div className="absolute bottom-4 right-4 text-xs text-gray-500 bg-gray-900/80 px-3 py-2 rounded-lg">
        <p>🖱️ Drag to rotate • Scroll to zoom • Shift+drag to pan</p>
      </div>
    </div>
  )
}
