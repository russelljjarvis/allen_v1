using HDF5
using Plots
using OnlineStats
using InteractiveUtils
using ThreadsX
using Plots
using Markdown
using Plots
using JLD2
using SpikeSynchrony
using LinearAlgebra
#unicodeplots()

using ColorSchemes

using Plots; gr()
using AngleBetweenVectors

function divide_epoch(nodes,times,sw,toi)
    t1=[]
    n1=[]
    t0=[]
    n0=[]
    @assert sw< toi
    third = toi-sw
    for (n,t) in zip(nodes,times)
        if sw<=t && t<toi
            append!(t0,t)
            append!(n0,n+1)            
        elseif t>=toi && t<=toi+third
            append!(t1,abs(t-toi))
            @assert t-toi>=0
            append!(n1,n+1)
        end
    #@assert length(t0) != 0 
    #@show(length(t0))
    #@show(length(t1))

    end
    (t0,n0,t1,n1)
end


function get_vector_coords(nodes,t0,n0,t1,n1)#times,duration)
    self_distances = Array{Float32}(zeros(maximum(nodes)+1))#([ for i in 0:maximum(nodes)])

    neuron0 =  Array{}([Float32[] for i in 0:maximum(nodes)])
    neuron1 =  Array{}([Float32[] for i in 0:maximum(nodes)])
    for (neuron,t) in zip(n0,t0)
        append!(neuron0[neuron],t)
    end
    for (neuron,t) in zip(n1,t1)
        append!(neuron1[neuron],t)
    end
    for (ind,(nt0,nt1)) in enumerate(zip(neuron0,neuron1))

        if length(nt0) == 0
            append!(nt0,0.0)
        end
        if length(nt1) == 0
            append!(nt1,0.0)
        end

        pooledspikes = vcat(nt0,nt1)
        maxt = maximum(sort!(unique(pooledspikes)))
        t1_ = sort(nt1)
        t0_ = sort(nt0)
        #for (tx,ty) in zip(t0_,t1_)
        #    @show(tx,ty)
        #end
        t, S = SPIKE_distance_profile(t0_,t1_;t0=0,tf = maxt)
        #@show(sum(S))
        #@show(sum(t))
        #@show(sum(S))

        self_distances[ind]=sum(S)
        #@show(t,S)
        #end
    end
    return self_distances
end

function get_()
    hf5 = h5open("spikes.h5","r")
    nodes = Vector{Int64}(read(hf5["spikes"]["v1"]["node_ids"]))
    #nodes2 = Vector{Int64}([i+maximum(nodes) for i in nodes])
    #nodes =  Vector{Int64}(reduce(hcat,(nodes, nodes2)))#,dims=1)#[nodes; nodes]#vcat(nodes,nodes)
    times = Vector{Float64}(read(hf5["spikes"]["v1"]["timestamps"]))
    #times = [times; times]
    close(hf5)
    @save "data_augmented_spikes.jld2" nodes times
    return (times,nodes)
end

using AngleBetweenVectors

function get_plot()
    times,nodes = get_()


    division_size = 10

    step_size = maximum(times)/division_size
    cnt = 1
    iters = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(iters)

    start_windows = collect(0:step_size:step_size*division_size-1)
    cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))
    p1 = plot()#title = "Plot 1")
    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
	PP = []
    for (ind,toi) in enumerate(iters)
        sw = start_windows[ind]

        (t0,n0,t1,n1) = divide_epoch(nodes,times,sw,toi)
        self_distances = get_vector_coords(nodes,t0,n0,t1,n1)
        mat_of_distances[ind,:] = self_distances



        o1 = HeatMap(zip(minimum(t0):maximum(t0)/1000.0:maximum(t0),minimum(n0):maximum(n0/1000.0):maximum(n0)) )
        fit!(o1,zip(t0,convert(Vector{Float64},n0)))
        p0 = plot(o1, marginals=false, legend=false) 
        
        if maximum(n1)/1000.0 >0
            if maximum(t1)/1000.0 >0
                #t1 = t1.-toi
                #t0 = t0.-toi
                o2 = HeatMap(zip(minimum(t1):maximum(t1)/1000.0:maximum(t1),minimum(n1):maximum(n1/1000.0):maximum(n1)) )
                fit!(o2,zip(t1,convert(Vector{Float64},n1)))
                p1 = plot(02, marginals=false, legend=false)
            end
        end
        push!(PP,p0)
        push!(PP,p1)



    end
    n = length(iters)
    plot(PP...; marginals=false, legend=false, size = default(:size) .* (n, 4), layout = (length(PP),1), left_margin = 5Plots.mm) |> display
    normalize!(mat_of_distances[:,:])
    #p=nothing
    #PP = []
    for (ind,self_distances) in enumerate(eachrow(mat_of_distances))
        #n = length(mat_of_distances[ind,:])
        #θ = LinRange(0, 2pi, n)
        #p = plot(θ, mat_of_distances[ind,:], proj=:polar,color=cs1[ind])#, layout = length(mat_of_distances))
        #push!(PP,plot!(p,θ,mat_of_distances[ind,:], proj=:polar,color=cs1[ind]))
        if ind>1
            @show(angle(mat_of_distances[ind,:],mat_of_distances[ind-1,:]))
        end

        ind+=1 
    end
	#plot(PP...; layout = (length(mat_of_distances), 1))|>display

    p=nothing
    #PP = []
    for (ind,self_distances) in enumerate(eachrow(mat_of_distances))
        n = length(mat_of_distances[ind,:])
        θ = LinRange(0, 2pi, n)
        if ind==1
            p = plot(θ, mat_of_distances[ind,:], proj=:polar,color=cs1[ind])#, layout = length(mat_of_distances))
        else 
            plot!(p,θ,mat_of_distances[ind,:], proj=:polar,color=cs1[ind])
        end

        ind+=1 
    end
    #current() |>display


    #plot!(plots_)|>display

    #@show(self_distances)
    if 1==0
        @time o = IndexedPartition(Float64, KHist(100), 100)
        @time fit!(o,zip(convert(Vector{Float64},nodes),times))
        @time plot(o) |> display
    end
    return mat_of_distances



end
@time mat_of_distances = get_plot()
#println("compiled times")
#@time get_plot()
#o = fit!(IndexedPartition(Float64, KHist(40), 40), zip(x, y))
#using Plots

#xy = zip(randn(10^6), randn(10^6))
#xy = zip(1 .+ randn(10^6) ./ 10, randn(10^6))
#0:Float64(maximum(nodes))/10.0:maximum(nodes)), 0.0:maximum(times)/100.0:maximum(times), xy)
#o1 = fit!(HeatMap(zip(1:maximum(nodes)/10.0:maximum(nodes)),0.0:maximum(times)/100.0:maximum(times)), xy)
#plot(o)
#=
#plot(o1)
function getstuff(times,nodes,o,o1)
    tempx = []
    tempy = []
    done = 0

    for ci=1:length(nodes) 

        push!(tempx, times[ci])
        push!(tempy, ci)
        if ci%10000==0

            convtempy = convert(Vector{Float64},tempy)
            store_conv_tempy = zip(convtempy, tempx)
            for (i,j) in store_conv_tempy
                fit!(o,zip(i,j))
                fit!(o1,zip(i,j))

            end
            #if 
            #fit!(o1,store_conv_tempy)
            #merge!(o1, o1)

            plot(o1, marginals=false, legend=true) |> display

            #fit!(o1,zip(convert(Vector{Float64},tempy), tempx))
            #plot(o1, marginals=false, legend=true) |> display
            plot(o) |> display
            #plot(ox) |> display       
            #tempx = []
            #tempy = []
            
        end

    end
    return tempx,tempy
end
println("done")
tempx,tempy = getstuff(times,nodes,o,o1)
println("done")
=#


#=
o = CCIPCA(2, length(unique(nodes))) 
fit!(o, zip(xs, ys))
OnlineStats.transform(o, zip(xs, ys))     # Project u3 into PCA space fitted to u1 and u2 but don't change the projection
u4 = rand(10)
OnlineStats.fittransform!(o, zip(xs, ys)) # Fit u4 and then project u4 into the space
sort!(o)                         # Sort from high to low eigenvalues
@show(OnlineStats.relativevariances(o))   
=#