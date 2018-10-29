using Images  # Basic image processing functions
using PyPlot  # Plotting and image loading
using FileIO  # Functions for loading and storing data in the ".jld2" format


# Load the image from the provided .jld2 file
function loaddata()

  return data::Array{Float64,2}
end


# Separate the image data into three images (one for each color channel),
# filling up all unknown values with 0
function separatechannels(data::Array{Float64,2})

  return r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2}
end


# Combine three color channels into a single image
function makeimage(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})

  return image::Array{Float64,3}
end


# Interpolate missing color values using bilinear interpolation
function interpolate(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})

  return image::Array{Float64,3}
end


# Display two images in a single figure window
function displayimages(img1::Array{Float64,3}, img2::Array{Float64,3})

end

#= Problem 2
Bayer Interpolation =#

function problem2()
  # load raw data
  data = loaddata()
  # separate data
  r,g,b = separatechannels(data)
  # merge raw pattern
  img1 = makeimage(r,g,b)
  # interpolate
  img2 = interpolate(r,g,b)
  # display images
  displayimages(img1, img2)
  return
end
