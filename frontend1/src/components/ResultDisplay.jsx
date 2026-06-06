import React from 'react';
import { CheckCircle, AlertCircle, AlertTriangle, Brain } from 'lucide-react';

function ResultDisplay({ result, confidence, lesionImage, lesionDetected }) {
  const getResultConfig = () => {
    switch (result) {
      case 'No Stroke':
        return {
          icon: <CheckCircle className="w-16 h-16" />,
          bgGradient: 'from-green-50 to-emerald-50',
          borderColor: 'border-green-200',
          iconBg: 'bg-green-100',
          iconColor: 'text-green-600',
          titleColor: 'text-green-800',
          subtitleColor: 'text-green-600',
        };
      case 'Hemorrhagic Stroke':
        return {
          icon: <AlertCircle className="w-16 h-16" />,
          bgGradient: 'from-red-50 to-rose-50',
          borderColor: 'border-red-200',
          iconBg: 'bg-red-100',
          iconColor: 'text-red-600',
          titleColor: 'text-red-800',
          subtitleColor: 'text-red-600',
        };
      case 'Ischemic Stroke':
        return {
          icon: <AlertTriangle className="w-16 h-16" />,
          bgGradient: 'from-yellow-50 to-amber-50',
          borderColor: 'border-yellow-200',
          iconBg: 'bg-yellow-100',
          iconColor: 'text-yellow-600',
          titleColor: 'text-yellow-800',
          subtitleColor: 'text-yellow-600',
        };
      default:
        return {
          icon: <Brain className="w-16 h-16" />,
          bgGradient: 'from-slate-50 to-gray-50',
          borderColor: 'border-slate-200',
          iconBg: 'bg-slate-100',
          iconColor: 'text-slate-600',
          titleColor: 'text-slate-800',
          subtitleColor: 'text-slate-600',
        };
    }
  };

  const config = getResultConfig();
  const lesionImageUrl = lesionImage ? `http://localhost:8000/${lesionImage.replace(/\\/g, '/')}` : null;

  return (
    <div className={`bg-gradient-to-br ${config.bgGradient} rounded-2xl shadow-card border ${config.borderColor} p-6 sm:p-8`}>
      {/* Result Header */}
      <div className="flex flex-col sm:flex-row items-center gap-6 mb-6">
        <div className={`p-4 rounded-full ${config.iconBg}`}>
          <div className={config.iconColor}>
            {config.icon}
          </div>
        </div>
        
        <div className="text-center sm:text-left">
          <h2 className={`text-2xl sm:text-3xl font-bold ${config.titleColor} mb-2`}>
            {result}
          </h2>
          <div className="flex items-center justify-center sm:justify-start gap-2">
            <span className="text-slate-600">Confidence:</span>
            <span className={`text-xl font-bold ${config.iconColor}`}>
              {confidence.toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Description */}
      <div className="mb-6">
        {result === 'No Stroke' ? (
          <p className="text-slate-600 text-center sm:text-left bg-white/60 rounded-xl p-4 border border-slate-200">
            <CheckCircle className="w-5 h-5 inline-block mr-2 text-green-500" />
            No lesions detected. Brain scan appears normal.
          </p>
        ) : (
          <p className={`text-center sm:text-left ${config.subtitleColor} bg-white/60 rounded-xl p-4 border ${config.borderColor}`}>
            <AlertCircle className="w-5 h-5 inline-block mr-2" />
            Abnormal regions detected. Immediate medical attention recommended.
          </p>
        )}
      </div>

      {/* Lesion Image */}
      {lesionDetected && lesionImageUrl && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-slate-700 mb-3 text-center sm:text-left">
            Lesion Visualization
          </h3>
          <div className="bg-white rounded-xl p-4 border border-slate-200 shadow-sm">
            <img
              src={lesionImageUrl}
              alt="Lesion Detection"
              className="w-full max-h-80 object-contain rounded-lg"
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" viewBox="0 0 400 300"><rect fill="%23f1f5f9" width="400" height="300"/><text fill="%236b7280" font-family="sans-serif" font-size="16" x="50%" y="50%" text-anchor="middle" dominant-baseline="middle">Image not available</text></svg>';
              }}
            />
          </div>
        </div>
      )}

      {/* Warning for stroke cases */}
      {lesionDetected && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-xl">
          <p className="text-red-700 text-sm text-center">
            <strong>⚠️ Important:</strong> This is an AI-assisted analysis. Please consult with a qualified 
            healthcare professional for proper diagnosis and treatment. Early intervention is critical 
            for stroke patients.
          </p>
        </div>
      )}
    </div>
  );
}

export default ResultDisplay;

