document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const mriImage = document.getElementById('mri-image');

    if (fileInput) {
        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const img = new Image();
                    img.src = e.target.result;
                    img.onload = () => {
                        const canvas = document.createElement('canvas');
                        const ctx = canvas.getContext('2d');
                        canvas.width = 244;
                        canvas.height = 244;
                        ctx.drawImage(img, 0, 0, 244, 244);
                        const imageData = ctx.getImageData(0, 0, 244, 244);
                        const data = imageData.data;
                        for (let i = 0; i < data.length; i += 4) {
                            const grayscale = data[i] * 0.3 + data[i + 1] * 0.59 + data[i + 2] * 0.11;
                            data[i] = data[i + 1] = data[i + 2] = grayscale;
                        }
                        ctx.putImageData(imageData, 0, 0);
                        mriImage.src = canvas.toDataURL();
                    };
                };
                reader.readAsDataURL(file);
            }
        });
    }
});
