/**
 * Drag & Drop File Upload Component
 *
 * Reusable drop-zone that wraps an existing <input type="file">.
 * Provides drag-over highlight, file validation, and optional thumbnail preview.
 */

const DEFAULT_ACCEPT = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'];
const DEFAULT_MAX_SIZE_MB = 50;

/**
 * @param {Object} options
 * @param {string} options.dropZoneId   - Drop area element ID
 * @param {string} options.fileInputId  - Hidden file input ID
 * @param {string} options.displayId    - Filename display element ID
 * @param {string} [options.previewId]  - Optional preview container ID
 * @param {string[]} [options.accept]   - Allowed extensions
 * @param {number} [options.maxSizeMB]  - Max file size in MB
 * @param {Function} [options.onFile]   - Callback (file) => void
 * @param {Function} [options.onError]  - Error callback (message) => void
 */
export function createDragDropUpload(options) {
    const {
        dropZoneId,
        fileInputId,
        displayId,
        previewId,
        accept = DEFAULT_ACCEPT,
        maxSizeMB = DEFAULT_MAX_SIZE_MB,
        onFile,
        onError,
    } = options;

    const zone = document.getElementById(dropZoneId);
    const input = document.getElementById(fileInputId);
    const display = document.getElementById(displayId);
    const preview = previewId ? document.getElementById(previewId) : null;

    if (!zone || !input) return;

    function validateFile(file) {
        if (!file) return 'No file selected';
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!accept.includes(ext)) {
            return `Invalid file type: ${ext}. Allowed: ${accept.join(', ')}`;
        }
        if (file.size > maxSizeMB * 1024 * 1024) {
            return `File too large (max ${maxSizeMB}MB)`;
        }
        return null;
    }

    function handleFile(file) {
        const err = validateFile(file);
        if (err) {
            if (onError) onError(err);
            return;
        }

        if (display) {
            display.textContent = file.name;
        }
        zone.classList.add('has-file');

        if (preview && file.type.startsWith('image/')) {
            preview.classList.remove('hidden');
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
            };
            reader.readAsDataURL(file);
        }

        if (onFile) onFile(file);
    }

    // Click to open file dialog
    zone.addEventListener('click', (e) => {
        if (e.target === input) return;
        input.click();
    });

    // Keyboard: Enter/Space to open
    zone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            input.click();
        }
    });

    // File input change
    input.addEventListener('change', () => {
        const file = input.files?.[0];
        if (file) handleFile(file);
    });

    // Drag events
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.remove('drag-over');

        const file = e.dataTransfer?.files?.[0];
        if (file) {
            // Sync the file to the input for FormData compatibility
            const dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;
            handleFile(file);
        }
    });

    return {
        reset() {
            input.value = '';
            if (display) display.textContent = zone.dataset.placeholder || 'Click or drag image here';
            zone.classList.remove('has-file');
            if (preview) {
                preview.innerHTML = '';
                preview.classList.add('hidden');
            }
        },
        getFile() {
            return input.files?.[0] || null;
        },
    };
}
