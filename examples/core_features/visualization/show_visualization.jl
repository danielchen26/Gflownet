# GFlowNet Visualization - Simple Runner
# Just run: julia show_visualization.jl

using Pkg
Pkg.activate(@__DIR__)

println("""
🎨 GFlowNet Beautiful Visualization
══════════════════════════════════════

Starting visualization system...
""")

# Paths
const VIZ_ROOT = joinpath(@__DIR__, "..", "..", "..", "src", "utils", "visualization")
const API_SERVER = joinpath(VIZ_ROOT, "api", "gflownet_server.jl") 
const WEB_DIR = joinpath(VIZ_ROOT, "web")

# Clean up any existing servers
println("🧹 Cleaning up...")
# Suppress output and errors from pkill when no processes are found
try run(pipeline(`pkill -f "node.*vite"`, devnull), wait=false) catch end
try run(pipeline(`pkill -f "julia.*gflownet_server"`, devnull), wait=false) catch end
sleep(2)  # Give more time for cleanup

# Start API server in background with proper environment
println("🚀 Starting API server...")
api_cmd = Cmd(`julia --project=$(@__DIR__) $API_SERVER`, dir=@__DIR__)
api_task = @async begin
    try
        run(api_cmd)
    catch e
        @error "API server crashed!" exception=e
    end
end

# Wait for API to be ready with better error checking
println("⏳ Waiting for API...")
global api_ready = false
for i in 1:30  # Give it more time
    try
        # Use HTTP.jl directly for health check
        using HTTP
        response = HTTP.get("http://localhost:8080/health", readtimeout=2)
        if response.status == 200
            println("✅ API server ready!")
            global api_ready = true
            break
        end
    catch e
        if i % 5 == 0
            println("   Still waiting... ($i/30)")
        end
        sleep(1)
    end
end

if !api_ready
    error("API server failed to start after 30 seconds. Check the logs above.")
end

# Install npm dependencies if needed
if !isdir(joinpath(WEB_DIR, "node_modules"))
    println("📦 Installing web dependencies (first time only)...")
    cd(WEB_DIR) do
        run(`npm install`)
    end
end

# Start web server
println("✨ Starting web dashboard...")
web_task = @async begin
    cd(WEB_DIR) do
        try
            run(`npm run dev`)
        catch e
            @warn "Web server stopped" exception=e
        end
    end
end

# Wait for web server
sleep(3)

# Open browser
println("🌐 Opening browser...")
if Sys.isapple()
    run(`open http://localhost:3000`)
elseif Sys.islinux()
    try run(`xdg-open http://localhost:3000`) catch end
elseif Sys.iswindows()
    run(`cmd /c start http://localhost:3000`)
end

println("""

✅ Visualization is running!

📍 View at: http://localhost:3000

🎮 Features:
• 3D trajectories with WebGL effects
• Real-time training dashboard
• Interactive flow field visualization
• Beautiful dark theme with neon colors

Press Ctrl+C to stop everything.
""")

# Wait for interrupt
try
    wait(api_task)
catch e
    if !isa(e, InterruptException)
        rethrow(e)
    end
finally
    println("\n👋 Stopping servers...")
    # Kill all processes (suppress output)
    try run(pipeline(`pkill -f "node.*vite"`, devnull)) catch end
    try run(pipeline(`pkill -f "julia.*gflownet_server"`, devnull)) catch end
    println("✅ Done!")
end