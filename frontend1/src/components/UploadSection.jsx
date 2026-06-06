import React, { useCallback, useState, useRef } from 'react';
import { Upload, X, FileText, Image } from 'lucide-react';

function UploadSection({ icon, title, description, accept, preview, onUpload, fileType }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onUpload(files[0]);
    }
  }, [onUpload]);

  const handleFileChange = useCallback((e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onUpload(files[0]);
    }
  }, [onUpload]);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemove = (e) => {
    e.stopPropagation();
    onUpload(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <div className="p-3 bg-gradient-to-br from-blue-100 to-purple-100 rounded-xl">
          {icon}
        </div>
        <div>
          <h3 className="text-lg font-semibold text-slate-800">{title}</h3>
          <p className="text-sm text-slate-500">{description}</p>
        </div>
      </div>

      {/* Upload Area */}
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          flex-1 min-h-[200px] rounded-xl border-2 border-dashed transition-all duration-300 cursor-pointer
          flex flex-col items-center justify-center p-6
          ${isDragging 
            ? 'border-blue-500 bg-blue-50' 
            : preview 
              ? 'border-green-300 bg-green-50' 
              : 'border-slate-300 hover:border-blue-400 hover:bg-blue-50/50'
          }
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          onChange={handleFileChange}
          className="hidden"
        />

        {preview ? (
          <div className="w-full h-full flex flex-col items-center">
            {fileType === 'image' && typeof preview === 'string' ? (
              <div className="relative w-full">
                <img
                  src={preview}
                  alt="CT Scan Preview"
                  className="max-h-40 mx-auto rounded-lg object-contain"
                />
                <button
                  onClick={handleRemove}
                  className="absolute top-0 right-0 p-1 bg-red-500 text-white rounded-full 
                           hover:bg-red-600 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                {fileType === 'excel' ? (
                  <FileText className="w-12 h-12 text-green-500" />
                ) : (
                  <Image className="w-12 h-12 text-green-500" />
                )}
                <div className="text-center">
                  <p className="font-medium text-slate-700 truncate max-w-[200px]">{preview}</p>
                  <p className="text-sm text-green-600">File uploaded</p>
                </div>
                <button
                  onClick={handleRemove}
                  className="p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>
        ) : (
          <>
            <div className={`
              p-4 rounded-full mb-4 transition-colors
              ${isDragging ? 'bg-blue-100' : 'bg-slate-100'}
            `}>
              <Upload className={`w-8 h-8 ${isDragging ? 'text-blue-500' : 'text-slate-400'}`} />
            </div>
            <p className="text-slate-600 font-medium mb-1">
              {isDragging ? 'Drop file here' : 'Drag & drop or click to upload'}
            </p>
            <p className="text-slate-400 text-sm">
              {accept === 'image/*' ? 'PNG, JPG, JPEG' : 'Excel files (.xlsx, .xls)'}
            </p>
          </>
        )}
      </div>
    </div>
  );
}

export default UploadSection;

