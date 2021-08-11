import numpy as np
import matplotlib.pyplot as plt
import gryds
import SimpleITK as sitk
import os
import random

# procedural deformations with gryds and DIR with SimpleElastix
batch_size = 1000
num_batches = 10
num_patches_per_registration = 5
ct_images = os.listdir('./dataset/')[0:70] # first 70 used for training

for batch in range(6, 6 + num_batches):
    X = np.zeros((batch_size, 64, 64, 4), dtype='float32')
    E = np.zeros((batch_size,), dtype='float32')
    random.shuffle(ct_images)
    total_generated = 0
    while total_generated < batch_size:
        for file in ct_images:
            volume = np.load('./dataset/' + file)
            slices = volume.shape[0]
            for slice in range(slices):

                (x_corner, y_corner) = np.random.randint(0, 211, size=2)
                image = volume[slice, x_corner:x_corner+300, y_corner:y_corner+300]


                # random deformation
                random_grid = np.random.rand(2, 3, 3)
                random_grid -= 0.5
                random_grid /= 7.5

                bspline = gryds.BSplineTransformation(random_grid)
                interpolator = gryds.Interpolator(image)
                transformed_image = interpolator.transform(bspline)

                a_grid = gryds.Grid(image.shape)
                transformed_grid = a_grid.transform(bspline)
                true_deformation = transformed_grid.grid - a_grid.grid

                # deformable registration
                fixedImage = sitk.GetImageFromArray(transformed_image)
                movingImage = sitk.GetImageFromArray(image)
                elastixImageFilter = sitk.ElastixImageFilter()
                elastixImageFilter.SetFixedImage(fixedImage)
                elastixImageFilter.SetMovingImage(movingImage)
                parameterMapVector = sitk.VectorOfParameterMap()
                parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
                elastixImageFilter.SetParameterMap(parameterMapVector)
                elastixImageFilter.Execute()
                dir_image = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()).reshape(300, 300)

                transformParameterMap = elastixImageFilter.GetTransformParameterMap()
                strx = sitk.TransformixImageFilter()
                strx.SetTransformParameterMap(transformParameterMap)
                strx.SetMovingImage(movingImage)
                strx.ComputeDeformationFieldOn()
                strx.Execute()
                registration_deformation = sitk.GetArrayFromImage(strx.GetDeformationField())

                # get patches and errors
                normed_dvf1 = (np.flip(registration_deformation, 2) - np.amin(registration_deformation)) / (
                            np.amax(registration_deformation) - np.amin(registration_deformation))
                normed_dvf2 = (np.moveaxis(true_deformation, 0, 2) - np.amin(true_deformation)) / (
                            np.amax(true_deformation) - np.amin(true_deformation))

                for _ in range(num_patches_per_registration):
                    (xx_corner, yy_corner) = np.random.randint(0, 235, size=2)
                    x = np.array([image[xx_corner:xx_corner+64, yy_corner:yy_corner+64],
                                  dir_image[xx_corner:xx_corner+64, yy_corner:yy_corner+64],
                                  normed_dvf1[xx_corner:xx_corner+64, yy_corner:yy_corner+64, 1],
                                  normed_dvf1[xx_corner:xx_corner+64, yy_corner:yy_corner+64, 0]
                                  ]
                                 )
                    x = np.moveaxis(x, 0, 2)
                    e = ((normed_dvf1[xx_corner:xx_corner + 64, yy_corner:yy_corner + 64] - normed_dvf2[xx_corner:xx_corner + 64, yy_corner:yy_corner + 64]) ** 2).mean()
                    try:
                        X[total_generated] = x
                        E[total_generated] = e
                    except:
                        pass
                    total_generated += 1
                np.save('./npydata/x/' + str(batch) + '.npy', X)
                np.save('./npydata/e/' + str(batch) + '.npy', E)
                if total_generated > batch_size:
                    break
            if total_generated > batch_size:
                break



