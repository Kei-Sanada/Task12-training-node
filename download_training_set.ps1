# ダウンロードするURL
$url = "https://fed-ledger-prod-dataset.s3.amazonaws.com/12/training_set.jsonl?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIASSFQ745NLT5K57N2%2F20250720%2Fus-east-2%2Fs3%2Faws4_request&X-Amz-Date=20250720T140547Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=7dc23ca4fdb2d9ee193f5ac6c301fcc095c1b9439b7c926be54becf17fd68f17"

# 保存先のファイル名
$output = "training_set.jsonl"

try {
    Write-Host "ダウンロードを開始します..."
    Invoke-WebRequest -Uri $url -OutFile $output
    Write-Host "ダウンロード完了: $output"
} catch {
    Write-Host "エラーが発生しました: $_"
}

Read-Host "Enterキーを押して終了します。"