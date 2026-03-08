# Verify Docker backend is up. Run from project root after: docker compose up --build -d
$base = "http://localhost:8000"
Write-Host "Checking backend at $base ..."
try {
    $r = Invoke-WebRequest -Uri "$base/docs" -UseBasicParsing -TimeoutSec 5
    if ($r.StatusCode -eq 200) {
        Write-Host "OK – Backend is up. Open: $base/docs (API docs), $base (redirect)"
        Write-Host "Frontend: open frontend/index.html in browser (API base: $base)"
        exit 0
    }
} catch {
    Write-Host "Backend not responding. Run: docker compose up --build -d (first build may take several minutes)."
    exit 1
}
