using PyPlot
using FileIO
using JLD2

# load and return the given image
function loadimage()
  img = PyPlot.imread("a1p1.png")
  return img::Array{Float32,3}
end

# save the image as a .jld2 file
function savefile(img::Array{Float32,3})
  save("data1.jld2","data",img)
end

# load and return the .jld2 file
function loadfile()
  img = load("data1.jld2","data")
  return img::Array{Float32,3}
end

# create and return a horizontally mirrored image
function mirrorhorizontal(img::Array{Float32,3})
  mirrored = img[:,end:-1:1,:]
  return mirrored::Array{Float32,3}
end

# display the normal and the mirrored image in one plot
function showimages(img1::Array{Float32,3}, img2::Array{Float32,3})
  figure()
  subplot(121), imshow(img1), title("normal image")
  subplot(122), imshow(img2), title("mirrored image")
end

#= Problem 1
Load and Display =#

function problem1()
  img1 = loadimage()
  savefile(img1)
  img2 = loadfile()
  img2 = mirrorhorizontal(img2)
  showimages(img1, img2)
end
