document.addEventListener("DOMContentLoaded", () => {
    // --- Upload Page Logic ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const selectedFilesDiv = document.getElementById('selected-files');
    const uploadBtn = document.getElementById('upload-btn');
    
    let selectedFiles = [];
    
    if (dropZone && fileInput) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight drop zone
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', handleDrop, false);
        
        // Handle file input selection
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
        
        uploadBtn.addEventListener('click', uploadFiles);
    }
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    function handleFiles(files) {
        files = [...files];
        selectedFiles = selectedFiles.concat(files);
        updateFileDisplay();
    }
    
    function updateFileDisplay() {
        if(!selectedFilesDiv) return;
        
        selectedFilesDiv.innerHTML = '';
        selectedFiles.forEach(file => {
            const chip = document.createElement('div');
            chip.className = 'file-chip';
            chip.textContent = file.name;
            selectedFilesDiv.appendChild(chip);
        });
        
        uploadBtn.disabled = selectedFiles.length === 0;
    }
    
    function uploadFiles() {
        if(selectedFiles.length === 0) return;
        
        const flightId = document.getElementById('flight_id').value || 'flight_' + new Date().getTime();
        
        const formData = new FormData();
        formData.append('flight_id', flightId);
        
        selectedFiles.forEach(file => {
            formData.append('images', file);
        });
        
        const progressDiv = document.getElementById('upload-progress');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        progressDiv.classList.remove('hidden');
        uploadBtn.disabled = true;
        
        // Simulate progress for UI (since fetch doesn't easily report upload progress without XMLHttpRequest)
        let simProgress = 0;
        const interval = setInterval(() => {
            simProgress += 5;
            if(simProgress > 90) clearInterval(interval);
            progressFill.style.width = simProgress + '%';
            progressText.textContent = `Uploading ${selectedFiles.length} files...`;
        }, 100);
        
        let currentFlightId = '';
        
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            clearInterval(interval);
            progressFill.style.width = '100%';
            progressText.textContent = `Pre-filtered ${data.processed} images!`;
            
            currentFlightId = data.flight_id;
            
            setTimeout(() => {
                // Hide upload section, show review section
                document.querySelector('.upload-form').classList.add('hidden');
                document.getElementById('upload-progress').classList.add('hidden');
                
                const reviewSection = document.getElementById('review-section');
                reviewSection.classList.remove('hidden');
                
                populateReviewGrids(data.results);
            }, 1000);
        })
        .catch(err => {
            clearInterval(interval);
            progressText.textContent = `Error uploading files.`;
            uploadBtn.disabled = false;
            console.error(err);
        });
        
        function populateReviewGrids(results) {
            const gridValid = document.getElementById('grid-valid');
            const gridInvalid = document.getElementById('grid-invalid');
            const countValid = document.getElementById('count-valid');
            const countInvalid = document.getElementById('count-invalid');
            
            gridValid.innerHTML = '';
            gridInvalid.innerHTML = '';
            
            let validList = [];
            let invalidList = [];
            
            results.forEach(item => {
                const img = document.createElement('img');
                img.src = `/static/uploads/${item.filename}`;
                img.className = 'thumbnail-item';
                img.dataset.filename = item.filename;
                img.title = "Click to move to other category";
                
                img.addEventListener('click', function() {
                    // Toggle lists
                    if (this.parentElement === gridValid) {
                        gridInvalid.appendChild(this);
                    } else {
                        gridValid.appendChild(this);
                    }
                    updateCounts();
                });
                
                if (item.is_leaf) {
                    gridValid.appendChild(img);
                } else {
                    gridInvalid.appendChild(img);
                }
            });
            
            updateCounts();
            
            function updateCounts() {
                countValid.textContent = gridValid.children.length;
                countInvalid.textContent = gridInvalid.children.length;
            }
        }
        
        document.getElementById('analyze-btn').addEventListener('click', function() {
            const btn = this;
            btn.disabled = true;
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running AI Analysis...';
            
            // Get all valid filenames
            const validGrid = document.getElementById('grid-valid');
            const filenames = Array.from(validGrid.children).map(img => img.dataset.filename);
            
            fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    flight_id: currentFlightId,
                    filenames: filenames
                })
            })
            .then(res => res.json())
            .then(data => {
                window.location.href = `/results?flight_id=${currentFlightId}`;
            })
            .catch(err => {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-microscope"></i> Confirm & Run AI Analysis';
                alert('Error running analysis');
                console.error(err);
            });
        });
    }
});
