export default function TumorStats({ tumorVolumes, volumeShape }) {
  const totalVoxels = volumeShape ? volumeShape[0] * volumeShape[1] * volumeShape[2] : 0
  
  const formatNumber = (num) => {
    return num.toLocaleString()
  }
  
  const getPercentage = (count) => {
    if (totalVoxels === 0) return '0.00'
    return ((count / totalVoxels) * 100).toFixed(2)
  }

  const stats = [
    {
      label: 'Necrotic Core',
      value: tumorVolumes?.necrotic_core || 0,
      color: 'from-red-500 to-red-600',
      bgColor: 'bg-red-500/20',
      textColor: 'text-red-400',
      icon: '🔴'
    },
    {
      label: 'Edema',
      value: tumorVolumes?.edema || 0,
      color: 'from-green-500 to-green-600',
      bgColor: 'bg-green-500/20',
      textColor: 'text-green-400',
      icon: '🟢'
    },
    {
      label: 'Enhancing Tumor',
      value: tumorVolumes?.enhancing_tumor || 0,
      color: 'from-yellow-500 to-yellow-600',
      bgColor: 'bg-yellow-500/20',
      textColor: 'text-yellow-400',
      icon: '🟡'
    },
    {
      label: 'Total Tumor',
      value: tumorVolumes?.total_tumor || 0,
      color: 'from-purple-500 to-pink-500',
      bgColor: 'bg-purple-500/20',
      textColor: 'text-purple-400',
      icon: '🧠'
    }
  ]

  return (
    <div className="glass rounded-2xl p-6">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
        <svg className="w-7 h-7 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        Segmentation Results
      </h2>

      {/* Volume Info */}
      {volumeShape && (
        <div className="mb-6 p-3 bg-gray-800/50 rounded-lg inline-block">
          <span className="text-sm text-gray-400">Volume Dimensions: </span>
          <span className="text-sm font-mono text-white">
            {volumeShape[0]} × {volumeShape[1]} × {volumeShape[2]} ({formatNumber(totalVoxels)} voxels)
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <div 
            key={index}
            className={`${stat.bgColor} rounded-xl p-4 border border-white/5`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">{stat.icon}</span>
              <span className="text-sm text-gray-300">{stat.label}</span>
            </div>
            <div className={`text-2xl font-bold ${stat.textColor}`}>
              {formatNumber(stat.value)}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {getPercentage(stat.value)}% of volume
            </div>
            
            {/* Mini progress bar */}
            <div className="mt-2 h-1 bg-gray-700 rounded-full overflow-hidden">
              <div 
                className={`h-full bg-gradient-to-r ${stat.color}`}
                style={{ width: `${Math.min(parseFloat(getPercentage(stat.value)) * 10, 100)}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
