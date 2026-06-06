import React from 'react';

function MetricsCard({ label, value, icon, color = 'blue' }) {
  const colorClasses = {
    blue: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      iconBg: 'bg-blue-100',
      iconColor: 'text-blue-600',
      textColor: 'text-blue-700',
    },
    purple: {
      bg: 'bg-purple-50',
      border: 'border-purple-200',
      iconBg: 'bg-purple-100',
      iconColor: 'text-purple-600',
      textColor: 'text-purple-700',
    },
    green: {
      bg: 'bg-green-50',
      border: 'border-green-200',
      iconBg: 'bg-green-100',
      iconColor: 'text-green-600',
      textColor: 'text-green-700',
    },
    yellow: {
      bg: 'bg-yellow-50',
      border: 'border-yellow-200',
      iconBg: 'bg-yellow-100',
      iconColor: 'text-yellow-600',
      textColor: 'text-yellow-700',
    },
  };

  const colors = colorClasses[color] || colorClasses.blue;

  return (
    <div className={`${colors.bg} ${colors.border} border rounded-xl p-5 shadow-sm card-hover`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-slate-600 font-medium text-sm uppercase tracking-wide">
          {label}
        </span>
        <div className={`p-2 rounded-lg ${colors.iconBg}`}>
          <div className={colors.iconColor}>
            {icon}
          </div>
        </div>
      </div>
      
      <div className="flex items-baseline gap-1">
        <span className={`text-3xl font-bold ${colors.textColor}`}>
          {typeof value === 'number' ? value.toFixed(1) : value}
        </span>
        <span className={`text-lg ${colors.textColor}`}>%</span>
      </div>
      
      {/* Progress bar */}
      <div className="mt-3 h-2 bg-white rounded-full overflow-hidden">
        <div 
          className={`h-full bg-gradient-to-r ${color === 'blue' ? 'from-blue-400 to-blue-600' : color === 'purple' ? 'from-purple-400 to-purple-600' : 'from-green-400 to-green-600'} rounded-full transition-all duration-1000`}
          style={{ width: `${Math.min(typeof value === 'number' ? value : parseFloat(value), 100)}%` }}
        />
      </div>
    </div>
  );
}

export default MetricsCard;

