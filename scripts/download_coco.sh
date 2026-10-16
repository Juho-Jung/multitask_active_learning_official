#!/bin/bash
# COCO dataset download script

COCO_DIR="./data/coco"
mkdir -p "$COCO_DIR"/{images,annotations}

cd "$COCO_DIR"

echo "📥 Starting COCO dataset download..."
echo "Save location: $COCO_DIR"
echo ""

# Download images
echo "=== Downloading images ==="
cd "$COCO_DIR/images"

echo "1. Downloading train2017.zip..."
if [ ! -f "train2017.zip" ]; then
    wget http://images.cocodataset.org/zips/train2017.zip
else
    echo "   train2017.zip already exists"
fi

echo "2. Downloading val2017.zip..."
if [ ! -f "val2017.zip" ]; then
    wget http://images.cocodataset.org/zips/val2017.zip
else
    echo "   val2017.zip already exists"
fi

echo "3. Downloading test2017.zip..."
if [ ! -f "test2017.zip" ]; then
    wget http://images.cocodataset.org/zips/test2017.zip
else
    echo "   test2017.zip already exists"
fi

echo "4. Downloading unlabeled2017.zip..."
if [ ! -f "unlabeled2017.zip" ]; then
    wget http://images.cocodataset.org/zips/unlabeled2017.zip
else
    echo "   unlabeled2017.zip already exists"
fi

# Download annotations
echo ""
echo "=== Downloading annotations ==="
cd "$COCO_DIR/annotations"

echo "1. Downloading annotations_trainval2017.zip..."
if [ ! -f "annotations_trainval2017.zip" ]; then
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
else
    echo "   annotations_trainval2017.zip already exists"
fi

echo "2. Downloading stuff_annotations_trainval2017.zip..."
if [ ! -f "stuff_annotations_trainval2017.zip" ]; then
    wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
else
    echo "   stuff_annotations_trainval2017.zip already exists"
fi

echo "3. Downloading image_info_test2017.zip..."
if [ ! -f "image_info_test2017.zip" ]; then
    wget http://images.cocodataset.org/annotations/image_info_test2017.zip
else
    echo "   image_info_test2017.zip already exists"
fi

echo "4. Downloading image_info_unlabeled2017.zip..."
if [ ! -f "image_info_unlabeled2017.zip" ]; then
    wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
else
    echo "   image_info_unlabeled2017.zip already exists"
fi

# Extract files
echo ""
echo "=== Extracting files ==="
cd "$COCO_DIR/images"

if [ -f "train2017.zip" ] && [ ! -d "train2017" ]; then
    echo "Extracting train2017.zip..."
    unzip -q train2017.zip
fi

if [ -f "val2017.zip" ] && [ ! -d "val2017" ]; then
    echo "Extracting val2017.zip..."
    unzip -q val2017.zip
fi

if [ -f "test2017.zip" ] && [ ! -d "test2017" ]; then
    echo "Extracting test2017.zip..."
    unzip -q test2017.zip
fi

if [ -f "unlabeled2017.zip" ] && [ ! -d "unlabeled2017" ]; then
    echo "Extracting unlabeled2017.zip..."
    unzip -q unlabeled2017.zip
fi

cd "$COCO_DIR/annotations"

if [ -f "annotations_trainval2017.zip" ] && [ ! -f "instances_train2017.json" ]; then
    echo "Extracting annotations_trainval2017.zip..."
    unzip -q annotations_trainval2017.zip
fi

if [ -f "stuff_annotations_trainval2017.zip" ] && [ ! -f "stuff_train2017.json" ]; then
    echo "Extracting stuff_annotations_trainval2017.zip..."
    unzip -q stuff_annotations_trainval2017.zip
fi

if [ -f "image_info_test2017.zip" ] && [ ! -f "image_info_test2017.json" ]; then
    echo "Extracting image_info_test2017.zip..."
    unzip -q image_info_test2017.zip
fi

if [ -f "image_info_unlabeled2017.zip" ] && [ ! -f "image_info_unlabeled2017.json" ]; then
    echo "Extracting image_info_unlabeled2017.zip..."
    unzip -q image_info_unlabeled2017.zip
fi

echo ""
echo "✅ Download and extraction complete!"
echo ""
echo "📁 Dataset structure:"
echo "   $COCO_DIR/images/train2017/ (images)"
echo "   $COCO_DIR/annotations/instances_train2017.json (annotations)"
echo ""
echo "🚀 Usage:"
echo "   python run_al.py --config configs/coco/coco_cls_det_seg.json"
echo "   or"
echo "   python run_al.py --dataset coco \\"
echo "     --coco_root $COCO_DIR/images \\"
echo "     --coco_ann_file $COCO_DIR/annotations/instances_train2017.json"
