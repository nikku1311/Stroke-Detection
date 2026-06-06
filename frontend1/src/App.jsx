import React, { useState, useCallback } from 'react';
import { Brain, Activity, Upload, AlertCircle, CheckCircle, AlertTriangle, RotateCcw, Loader2 } from 'lucide-react';
import UploadSection from './components/UploadSection';
import ResultDisplay from './components/ResultDisplay';
import MetricsCard from './components/MetricsCard';

const API_URL = 'http://localhost:8000';

function App() {
  const [ctImage, setCtImage] = useState(null);
  const [eegFile, setEegFile] = useState(null);
  const [ctPreview, setCtPreview] = useState(null);
  const [eegPreview, setEegPreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleCTUpload = useCallback((file) => {
    if (file) {
      setCtImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setCtPreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const handleEEGUpload = useCallback((file) => {
    if (file) {
      setEegFile(file);
      setEegPreview(file.name);
    }
  }, []);

  const handleAnalyze = async () => {
    if (!ctImage || !eegFile) {
      setError('Please upload both CT scan and EEG file');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('ct_image', ctImage);
    formData.append('eeg_file', eegFile);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'Failed to analyze. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setCtImage(null);
    setEegFile(null);
    setCtPreview(null);
    setEegPreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-sm border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col items-center text-center">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg">
                <Brain className="w-8 h-8 text-white" />
              </div>
            </div>
            <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              A Hybrid Approach for Detection and Classification of Brain Stroke 
              using Non-Contrast CT Imaging and EEG Time Series Data
            </h1>
            <p className="text-slate-500 mt-2 text-sm sm:text-base max-w-3xl">
              Advanced deep learning model combining 3D-ResNet18 and Graph Convolutional Network (GCN) 
              for accurate stroke detection from multimodal medical data.
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-3 animate-fade-in">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Upload Section */}
        {!results && (
          <div className="animate-fade-in">
            <div className="bg-white rounded-2xl shadow-card p-6 sm:p-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
                {/* CT Scan Upload */}
                <UploadSection
                  icon={<Brain className="w-10 h-10 text-blue-500" />}
                  title="CT Scan Image"
                  description="Upload non-contrast CT brain scan"
                  accept="image/*"
                  preview={ctPreview}
                  onUpload={handleCTUpload}
                  fileType="image"
                />

                {/* EEG Upload */}
                <UploadSection
                  icon={<Activity className="w-10 h-10 text-purple-500" />}
                  title="EEG Time Series Data"
                  description="Upload EEG data as Excel file (.xlsx)"
                  accept=".xlsx,.xls"
                  preview={eegPreview}
                  onUpload={handleEEGUpload}
                  fileType="excel"
                />
              </div>

              {/* Analyze Button */}
              <div className="mt-8 flex justify-center">
                <button
                  onClick={handleAnalyze}
                  disabled={!ctImage || !eegFile || isLoading}
                  className={`
                    flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-lg
                    transition-all duration-300 shadow-lg
                    ${ctImage && eegFile && !isLoading
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700 hover:shadow-xl hover:scale-105'
                      : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                    }
                  `}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Activity className="w-6 h-6" />
                      Analyze & Detect Stroke
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {results && (
          <div className="space-y-6 animate-fade-in">
            {/* Result Display */}
            <ResultDisplay
              result={results.result}
              confidence={results.confidence}
              lesionImage={results.lesion_image}
              lesionDetected={results.lesion_detected}
            />

            {/* Metrics Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <MetricsCard
                label="Precision"
                value={results.precision}
                icon={<Activity className="w-5 h-5" />}
                color="blue"
              />
              <MetricsCard
                label="Recall"
                value={results.recall}
                icon={<AlertCircle className="w-5 h-5" />}
                color="purple"
              />
              <MetricsCard
                label="F1 Score"
                value={results.f1_score * 100}
                icon={<CheckCircle className="w-5 h-5" />}
                color="green"
              />
            </div>

            {/* New Analysis Button */}
            <div className="flex justify-center pt-4">
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-6 py-3 bg-white border-2 border-slate-200 
                         rounded-xl font-semibold text-slate-700 hover:border-blue-400 hover:text-blue-600
                         transition-all duration-300 shadow-sm hover:shadow-md"
              >
                <RotateCcw className="w-5 h-5" />
                New Analysis
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 border-t border-slate-200 bg-white/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-slate-500 text-sm">
          <p>AI-Powered Stroke Detection System • For Medical Assistance Only</p>
          <p className="mt-1">Consult a healthcare professional for diagnosis</p>
        </div>
      </footer>
    </div>
  );
}

export default App;

