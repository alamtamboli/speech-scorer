import React, { useState } from "react";

export default function AudioUploader({ setResult }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const upload = async () => {
    if (!file) return alert("Please select an audio file");

    const fd = new FormData();
    fd.append("file", file);

    setLoading(true);

    try {
      const resp = await fetch("http://localhost:8000/upload_audio", {
        method: "POST",
        body: fd,
      });

      const data = await resp.json();
      setResult(data);
    } catch (err) {
      alert("Upload failed: " + err);
    }

    setLoading(false);
  };

  return (
    <div className="panel">
      <h4>Upload Audio</h4>
      <input type="file" accept="audio/*" onChange={(e) => setFile(e.target.files[0])} />

      <button onClick={upload} disabled={loading}>
        {loading ? "Processing..." : "Upload & Score"}
      </button>
    </div>
  );
}
