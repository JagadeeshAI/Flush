#!/bin/bash
# Create animation from t-SNE visualization frames

echo "Creating animation from frames..."

# Using ffmpeg to create MP4 animation
ffmpeg -y -framerate 2 -pattern_type glob -i "gray_viz/step_*.png" \
       -c:v libx264 -pix_fmt yuv420p -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2" \
       "gray_viz/voronoi_unlearning_animation.mp4"

# Create GIF version (optional)
ffmpeg -y -framerate 2 -pattern_type glob -i "gray_viz/step_*.png" \
       -vf "scale=800:600:force_original_aspect_ratio=decrease,pad=800:600:(ow-iw)/2:(oh-ih)/2,palettegen" \
       "gray_viz/palette.png"

ffmpeg -y -framerate 2 -pattern_type glob -i "gray_viz/step_*.png" \
       -i "gray_viz/palette.png" \
       -filter_complex "scale=800:600:force_original_aspect_ratio=decrease,pad=800:600:(ow-iw)/2:(oh-ih)/2[x];[x][1:v]paletteuse" \
       "gray_viz/voronoi_unlearning_animation.gif"

echo "Animation created: gray_viz/voronoi_unlearning_animation.mp4"
echo "GIF created: gray_viz/voronoi_unlearning_animation.gif"
