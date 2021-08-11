import numpy as np
import matplotlib.pyplot as plt
import gryds
import SimpleITK as sitk

# Define a random 3x3 B-spline grid for a 2D image:
random_grid = np.random.rand(2, 3, 3)
random_grid -= 0.5
random_grid /= 7.5

full_image = np.load('dataset/5.npy')
image = full_image[50, 150:450, 150:450]
# image += 0.2 * np./random.random((50, 50))

bspline = gryds.BSplineTransformation(random_grid)
interpolator = gryds.Interpolator(image)
transformed_image = interpolator.transform(bspline)

a_grid = gryds.Grid(image.shape)
transformed_grid = a_grid.transform(bspline)
displacement_field = transformed_grid.grid - a_grid.grid

fixedImage = sitk.GetImageFromArray(transformed_image)
movingImage = sitk.GetImageFromArray(image)
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixedImage)
elastixImageFilter.SetMovingImage(movingImage)
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
elastixImageFilter.SetParameterMap(parameterMapVector)
elastixImageFilter.Execute()
resultImage = sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()).reshape(300, 300)

transformParameterMap = elastixImageFilter.GetTransformParameterMap()
strx = sitk.TransformixImageFilter()
strx.SetTransformParameterMap(transformParameterMap)
strx.SetMovingImage(movingImage)
strx.ComputeDeformationFieldOn()
strx.Execute()
dvf = sitk.GetArrayFromImage(strx.GetDeformationField())

m1 = np.mean(np.abs(displacement_field[0]))
m2 = np.mean(np.abs(displacement_field[1]))
n1 = np.mean(np.abs(dvf[:, :, 0]))
n2 = np.mean(np.abs(dvf[:, :, 1]))

print(m1, m2, n1, n2)
print(m1 / n1, n2 / m2)


fig, axs = plt.subplots(2, 4)
axs[0, 0].imshow(image[100:200, 100:200])
axs[1, 0].imshow(transformed_image[100:200, 100:200])
axs[0, 1].imshow(displacement_field[0, 100:200, 100:200])
axs[1, 1].imshow(displacement_field[1, 100:200, 100:200])
axs[0, 2].imshow(image[100:200, 100:200])
axs[1, 2].imshow(resultImage[100:200, 100:200])
axs[0, 3].imshow(dvf[100:200, 100:200, 1])
axs[1, 3].imshow(dvf[100:200, 100:200, 0])
plt.show()


