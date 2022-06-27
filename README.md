# Deformable image registation anomaly detection dataset generator

A simple project to read DICOM images, apply random deformations to those 2D images with gryds splines, and learn the deformations according to a SimpleElastix deformable image registation (DIR) algorithm. Creates a dataset to be used in training a DIR error detection model based on predicted (SimpleElastix) and ground truth (B-splines) deformations.
