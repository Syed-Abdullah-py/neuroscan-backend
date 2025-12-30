import face_recognition
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, status, File, Form
from fastapi.middleware.cors import CORSMiddleware
import io
import json
from typing import List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_face_encoding_from_upload(file: UploadFile):
    """
    Reads an image file and returns the 128-dimensional face encoding.
    Returns None if no face is found.
    """
    try:
        contents = await file.read()
        # Reset cursor for subsequent reads if necessary
        await file.seek(0)
        
        image = face_recognition.load_image_file(io.BytesIO(contents))
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            return None
            
        return face_encodings[0]
    except Exception as e:
        print(f"[ERROR] Error processing face: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
    """
    Compares a known face encoding against a candidate encoding.
    """
    if known_encoding is None or unknown_encoding is None:
        print("[DEBUG] One of the encodings is None")
        return False
    
    # Convert to numpy arrays if needed
    known_encoding = np.array(known_encoding)
    unknown_encoding = np.array(unknown_encoding)
    
    # Calculate face distance for debugging
    face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    
    results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=tolerance)
    return results[0], face_distance

@app.post("/encode")
async def encode_endpoint(file: UploadFile = File(...)):
    encoding = await get_face_encoding_from_upload(file)
    if encoding is None:
        return {"error": "No face found"}
    
    # Return as list for JSON serialization
    return {"encoding": encoding.tolist()}

@app.post("/recognize")
async def recognize_endpoint(file: UploadFile = File(...), known_data: str = Form(...)):
    """
    known_data: JSON string of list of users: [{"userId": "...", "encoding": [float, ...]}, ...]
    """
    unknown_encoding = await get_face_encoding_from_upload(file)
    if unknown_encoding is None:
        return {"error": "No face found in input image"}
        
    try:
        known_users = json.loads(known_data)
    except:
        raise HTTPException(status_code=400, detail="Invalid known_data format")

    best_match_id = None
    best_distance = 1.0 # Lower is better
    
    for user in known_users:
        u_id = user.get("userId")
        u_enc = user.get("encoding")
        
        if not u_id or not u_enc:
            continue
            
        is_match, distance = compare_faces(u_enc, unknown_encoding)
        
        if is_match and distance < best_distance:
            best_distance = distance
            best_match_id = u_id
            
    if best_match_id:
        return {"match": True, "userId": best_match_id, "distance": best_distance}
    else:
        return {"match": False}

@app.post("/validate-case-files")
async def validate_case_files(files: List[UploadFile] = File(...)):
    """
    Validates uploaded files for a case.
    Rules:
    - If .h5 files: Count must be 155.
    - If .nii.gz files: Count must be 4.
    - If .zip / .tar: Extract temp and apply same rules to contents.
    
    Returns:
        JSON with "valid": boolean, "message": string, "file_count": int, "type": string
    """
    import tempfile
    import os
    import shutil
    import zipfile
    import  tarfile
    from pathlib import Path

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save uploaded files to temp dir
        saved_files = []
        for file in files:
            file_path = temp_path / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            
        # Check if archive
        is_archive = False
        if len(saved_files) == 1:
            f = saved_files[0]
            if f.suffix in ['.zip', '.tar', '.gz']: # Simple check, might need better MIME check
                is_archive = True
                # Extract
                extract_path = temp_path / "extracted"
                extract_path.mkdir()
                
                try:
                    if f.suffix == '.zip':
                        with zipfile.ZipFile(f, 'r') as zip_ref:
                            zip_ref.extractall(extract_path)
                    elif f.name.endswith('.tar.gz') or f.suffix == '.tar':
                        with tarfile.open(f, 'r:*') as tar_ref:
                            tar_ref.extractall(extract_path)
                except Exception as e:
                     return {"valid": False, "message": f"Failed to extract archive: {str(e)}"}

                # Update saved_files list to be the extracted files (scan recursively)
                saved_files = [p for p in extract_path.rglob("*") if p.is_file() and not p.name.startswith('.')]
        
        # Count and classify
        h5_count = 0
        nii_count = 0
        
        for f in saved_files:
            if f.name.endswith('.h5'):
                h5_count += 1
            elif f.name.endswith('.nii.gz'):
                nii_count += 1
                
        # Apply Logic
        if h5_count > 0 and nii_count > 0:
             return {"valid": False, "message": "Mixed file types found (.h5 and .nii.gz). Please upload only one type."}
        
        if h5_count > 0:
            if h5_count == 155:
                return {"valid": True, "message": "Valid .h5 dataset", "file_count": h5_count, "type": "h5"}
            else:
                 return {"valid": False, "message": f"Invalid .h5 file count. Expected 155, got {h5_count}", "file_count": h5_count}
                 
        if nii_count > 0:
            if nii_count == 4:
                return {"valid": True, "message": "Valid .nii.gz dataset", "file_count": nii_count, "type": "nii.gz"}
            else:
                return {"valid": False, "message": f"Invalid .nii.gz file count. Expected 4, got {nii_count}", "file_count": nii_count}
                
        return {"valid": False, "message": "No valid .h5 or .nii.gz files found in upload."}

