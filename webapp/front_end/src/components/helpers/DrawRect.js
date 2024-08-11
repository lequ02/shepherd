const DrawRect = (detections, ctx, canvasPosition) => {
    detections.forEach((prediction) => {
        const [x, y, width, height] = prediction["bbox"];
        const {
            left, right, top, bottom,
        } = canvasPosition;
        const text = prediction["class"];
        const color = Math.floor(Math.random() * 16777215).toString(16);
        ctx.strokeStyle = "#" + color;
        ctx.font = "18px Arial";

        ctx.beginPath();
        ctx.fillStyle = "#" + color;
        ctx.fillText(text, x, y);
        const rectX = (left <= x <= right) ? x : left;
        const rectY = (top <= y <= bottom) ? y : top;
        const rectWidth = rectX + width > right ? right - rectX : width;
        const rectHeight = rectY + height > bottom ? bottom - rectY : height;
        ctx.rect(
            rectX,
            rectY,
            rectWidth,
            rectHeight,
        );
        ctx.stroke();

        if (prediction.class === "car") {
        }
    });
};

export default DrawRect;
