param(
	[Parameter(Mandatory = $true, Position = 0)]
	[int]$NumGPUs,

	[Parameter(ValueFromRemainingArguments = $true, Position = 1)]
	[string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

function Run-Step {
	param(
		[Parameter(Mandatory=$true)][string]$Exe,
		[Parameter(Mandatory=$true)][string[]]$Arguments,
		[Parameter(Mandatory=$true)][string]$Desc
	)
	Write-Host "`n$Desc..." -ForegroundColor Cyan
	& $Exe @Arguments
	if ($LASTEXITCODE -ne 0) {
		throw "Failed: $Desc (exit $LASTEXITCODE)"
	}
	Write-Host "$Desc finished." -ForegroundColor Green
}

Write-Host "$NumGPUs GPUs configured" -ForegroundColor Yellow

# Preprocess
Run-Step -Exe 'python' -Arguments @('preprocess.py') + ($ExtraArgs ?? @()) -Desc 'Preprocessing'

# Split
Run-Step -Exe 'python' -Arguments @('split_dataset.py') + ($ExtraArgs ?? @()) -Desc 'Splitting dataset'

# Pack
Run-Step -Exe 'python' -Arguments @('pack_dataset.py') + ($ExtraArgs ?? @()) -Desc 'Packaging dataset'

# Train with torchrun (fallback to python -m torch.distributed.run if torchrun not found)
$torchrun = Get-Command 'torchrun' -ErrorAction SilentlyContinue
if ($null -ne $torchrun) {
	Write-Host "Starting training on $NumGPUs GPUs with torchrun.." -ForegroundColor Yellow
	Run-Step -Exe 'torchrun' -Arguments @('--nproc-per-node', "$NumGPUs", 'train.py') + ($ExtraArgs ?? @()) -Desc 'Training'
} else {
	Write-Host "torchrun not found. Falling back to 'python -m torch.distributed.run'" -ForegroundColor DarkYellow
	Run-Step -Exe 'python' -Arguments @('-m','torch.distributed.run','--nproc-per-node', "$NumGPUs", 'train.py') + ($ExtraArgs ?? @()) -Desc 'Training'
}

Write-Host "Training finished." -ForegroundColor Green
