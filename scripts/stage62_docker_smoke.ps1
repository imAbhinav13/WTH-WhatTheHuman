param(
    [string]$Image = "wth-api:stage6.2",
    [string]$Container = "wth-api-stage62",
    [int]$HostPort = 10000,
    [string]$EnvFile = ".env",
    [switch]$KeepContainer
)

$ErrorActionPreference = "Stop"

function Stop-WthContainer {
    $existing = docker ps -aq --filter "name=^/$Container$"
    if ($existing) {
        docker rm -f $Container | Out-Null
    }
}

Write-Host "Stage 6.2 Docker smoke test"
Write-Host "Project: $PWD"
Write-Host ""

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install/start Docker Desktop first."
}

docker version | Out-Null

if (-not (Test-Path "pyproject.toml")) {
    throw "Run this script from the WTH repository root."
}

if (-not (Test-Path "uv.lock")) {
    throw "uv.lock was not found in the repository root."
}

if (-not (Test-Path $EnvFile)) {
    throw "Environment file '$EnvFile' was not found."
}

Stop-WthContainer

try {
    Write-Host "[1/5] Building production image..."
    docker build `
        --pull `
        --tag $Image `
        .

    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed."
    }

    Write-Host "[2/5] Starting container..."
    docker run `
        --detach `
        --name $Container `
        --publish "${HostPort}:10000" `
        --env-file $EnvFile `
        --env PORT=10000 `
        $Image | Out-Null

    if ($LASTEXITCODE -ne 0) {
        throw "Container failed to start."
    }

    Write-Host "[3/5] Waiting for /api/health..."
    $healthy = $false

    for ($attempt = 1; $attempt -le 30; $attempt++) {
        Start-Sleep -Seconds 2

        try {
            $health = Invoke-RestMethod `
                -Uri "http://127.0.0.1:$HostPort/api/health" `
                -TimeoutSec 5

            if ($health.status -eq "healthy") {
                $healthy = $true
                break
            }
        }
        catch {
            # Startup may still be in progress.
        }
    }

    if (-not $healthy) {
        Write-Host ""
        Write-Host "Container logs:"
        docker logs $Container
        throw "Container did not become healthy."
    }

    Write-Host "PASS  /api/health"

    Write-Host "[4/5] Verifying public OpenAPI contract..."
    $schema = Invoke-RestMethod `
        -Uri "http://127.0.0.1:$HostPort/openapi.json" `
        -TimeoutSec 10

    $paths = @($schema.paths.PSObject.Properties.Name)

    $required = @(
        "/api/query",
        "/api/chunk/{chunk_id}",
        "/api/health",
        "/api/ready"
    )

    foreach ($path in $required) {
        if ($paths -notcontains $path) {
            throw "Missing public API path: $path"
        }
        Write-Host "PASS  $path"
    }

    $legacy = @($paths | Where-Object { $_ -like "/api/v1*" })
    if ($legacy.Count -gt 0) {
        throw "Legacy /api/v1 route found: $($legacy -join ', ')"
    }
    Write-Host "PASS  No /api/v1 routes"

    Write-Host "[5/5] Checking readiness..."
    try {
        $ready = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$HostPort/api/ready" `
            -TimeoutSec 30

        Write-Host "Readiness status: $($ready.status)"
    }
    catch {
        Write-Host "Readiness returned a non-2xx response."
        Write-Host "This does not invalidate the Docker image itself."
        Write-Host "Inspect configuration/dependency status before Stage 6.3."
    }

    Write-Host ""
    Write-Host "Stage 6.2 Docker smoke: PASS"
}
finally {
    if ($KeepContainer) {
        Write-Host "Container left running: $Container"
        Write-Host "URL: http://127.0.0.1:$HostPort"
    }
    else {
        Stop-WthContainer
        Write-Host "Container removed."
    }
}
