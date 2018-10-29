using Images  # Basic image processing functions
using PyPlot  # Plotting and image loading
using FileIO  # Functions for loading and storing data in the ".jld2" format


# Load the image from the provided .jld2 file
function loaddata()
  data = load("imagedata.jld2", "data")
  return data::Array{Float64,2}
end


# Separate the image data into three images (one for each color channel),
# filling up all unknown values with 0
function separatechannels(data::Array{Float64,2})
  r = zeros(size(data))
  g = zeros(size(data))
  b = zeros(size(data))
  r[1:2:end,1:2:end] = data[1:2:end,1:2:end]
  r[2:2:end,2:2:end] = data[2:2:end,2:2:end]
  g[1:2:end,2:2:end] = data[1:2:end,2:2:end]
  b[2:2:end,1:2:end] = data[2:2:end,1:2:end]
  return r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2}
end


# Combine three color channels into a single image
function makeimage(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})
  image = cat(r,g,b,dims=3)
  return image::Array{Float64,3}
end


# Interpolate missing color values using bilinear interpolation
function interpolate(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})
  # for red filter: each unknown pixel always has 4 known neighbors at sides (weighted 0.25)
  # for green/blue filter: each unknown pixel has either 2 known neighbors at sides (weighted 0.5)
  #                        or 4 known neighbors at corners (weighted 0.25)
  filter_r = [0 0.25 0; 0.25 1 0.25; 0 0.25 0]
  filter_g = [0.25 0.5 0.25; 0.5 1 0.5; 0.25 0.5 0.25]
  filter_b = [0.25 0.5 0.25; 0.5 1 0.5; 0.25 0.5 0.25]

  # boundary pixels are interpolated using values of the nearest neighbors (option "replicate")
  r_inter = imfilter(r,filter_r,"replicate")
  g_inter = imfilter(g,filter_g,"replicate")
  b_inter = imfilter(b,filter_b,"replicate")

  # combine interpolated color channels
  image = cat(r_inter,g_inter,b_inter,dims=3)
  return image::Array{Float64,3}
end


# Display two images in a single figure window
function displayimages(img1::Array{Float64,3}, img2::Array{Float64,3})
  figure()
  subplot(121), imshow(img1), title("RGB image")
  subplot(122), imshow(img2), title("interpolated RGB image")
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
