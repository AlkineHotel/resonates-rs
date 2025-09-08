function Swap-Amp {
	[CmdletBinding()]
	param (
		$Cargo = "$ENV:RESONATES_RS\Cargo.toml",
		$SwapString = "amp_main.rs",
		$SwapRegex = "amp_main(.*)`.rs",
		[switch]$ToAmped,
		[switch]$ToMain
		
	)
	if ([string]::IsNullOrEmpty($SwapRegex)) { $SwapRegex = $SwapString }
		$Cargo = "$ENV:RESONATES_RS\Cargo.toml"
	write-verbose "Checking for Cargo.toml"
	if (-not $(Test-Path $Cargo -errorAction SilentlyContinue)) {
		$Cargo = "E:\Users\alkenethiol\.vscode\ekfuncs\resonates-rs\Cargo.toml"
		if (-not $(Test-Path $Cargo -errorAction SilentlyContinue)) {
			Write-Error "Could not find Cargo.toml"
			return
		}
		else {write-verbose "Found Cargo.toml, but not in environent variable"}
	}
	else {write-verbose "Found Cargo.toml in environment variable"}
	$CargoBak = $("${Cargo}.bak_",$([datetime]::Now.tostring("yyyyMMdd_hhmm")) -join "")
	cp $Cargo -Destination $CargoBak && write-verbose "Backed up Cargo.toml tp ${CargoBak}"
	$CargoNow = get-content $Cargo -raw
	if ($ToAmped) {
	$CargoAmped = $CargoNow -replace "main.rs",$SwapString
	}
	elseif ($ToMain) {
		$CargoAmped = $CargoNow -replace $SwapRegex,"main.rs"
	}
	elseif ($CargoNow -match [regex]::escape($SwapString)) {
		$CargoAmped = $CargoNow -replace $SwapRegex,"main.rs"
		write-verbose "Swapped $SwapString with main.rs in $Cargo"
	}
	elseif ($CargoNow -match "src\/main`.rs") {
		$CargoAmped = $CargoNow -replace "main`.rs",$SwapString
		write-verbose "Swapped main.rs with amp_main.rs in $Cargo"
	}
	else {
		Write-Error "No switches provided, and no amped or main.rs found in Cargo.toml"
		return
	}

	set-content $Cargo $CargoAmped.trim()
	#write-verbose "Swapped main.rs with amp_main.rs in $Cargo"

}