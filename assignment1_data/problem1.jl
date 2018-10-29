using PyPlot
using FileIO

# load and return the given image
function loadimage()

  return img::Array{Float32,3}
end

# save the image as a .jld2 file
function savefile(img::Array{Float32,3})

end

# load and return the .jld2 file
function loadfile()

  return img::Array{Float32,3}
end

# create and return a horizontally mirrored image
function mirrorhorizontal(img::Array{Float32,3})

  return mirrored::Array{Float32,3}
end

# display the normal and the mirrored image in one plot
function showimages(img1::Array{Float32,3}, img2::Array{Float32,3})
  
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
