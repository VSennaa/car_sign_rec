# Define a sua rede e a porta RTSP padrão
$subnet = "10.0.0" # <-- ALTERE AQUI para a sua rede (ex: 192.168.0 ou 10.0.0)
$port = 554

Write-Host "Iniciando varredura rápida na rede $subnet.x buscando a porta $port (RTSP)..." -ForegroundColor Cyan

1..254 | ForEach-Object {
    $ip = "$subnet.$_"
    $tcp = New-Object System.Net.Sockets.TcpClient
    
    # Tenta conectar com um timeout curto (100ms) para ser bem rápido
    $async = $tcp.BeginConnect($ip, $port, $null, $null)
    $wait = $async.AsyncWaitHandle.WaitOne(100, $false)
    
    if ($wait) {
        try {
            $tcp.EndConnect($async)
            Write-Host "[+] Câmera RTSP encontrada no IP: $ip" -ForegroundColor Green
        } catch { 
            # Porta fechada ou erro, ignora silenciosamente
        }
    }
    $tcp.Close()
}

Write-Host "Varredura concluída!" -ForegroundColor Cyan