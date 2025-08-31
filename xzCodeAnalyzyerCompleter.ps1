using namespace System.Management.Automation
using namespace System.Management.Automation.Language

Register-ArgumentCompleter -Native -CommandName 'xzCodeAnalyzer' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    $commandElements = $commandAst.CommandElements
    $command = @(
        'xzCodeAnalyzer'
        for ($i = 1; $i -lt $commandElements.Count; $i++) {
            $element = $commandElements[$i]
            if ($element -isnot [StringConstantExpressionAst] -or
                $element.StringConstantType -ne [StringConstantType]::BareWord -or
                $element.Value.StartsWith('-') -or
                $element.Value -eq $wordToComplete) {
                break
        }
        $element.Value
    }) -join ';'

    $completions = @(switch ($command) {
        'xzCodeAnalyzer' {
            [CompletionResult]::new('-p', '-p', [CompletionResultType]::ParameterName, 'Path to analyze')
            [CompletionResult]::new('--path', '--path', [CompletionResultType]::ParameterName, 'Path to analyze')
            [CompletionResult]::new('--max-size', '--max-size', [CompletionResultType]::ParameterName, 'Maximum chunk size (characters)')        
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Output FOLDER for chunk analysis results')
            [CompletionResult]::new('--folder', '--folder', [CompletionResultType]::ParameterName, 'Output FOLDER for chunk analysis results')   
            [CompletionResult]::new('-o', '-o', [CompletionResultType]::ParameterName, 'Output file for chunk analysis results')
            [CompletionResult]::new('--output', '--output', [CompletionResultType]::ParameterName, 'Output file for chunk analysis results')     
            [CompletionResult]::new('-l', '-l', [CompletionResultType]::ParameterName, 'Language to analyze')
            [CompletionResult]::new('--language', '--language', [CompletionResultType]::ParameterName, 'Language to analyze')
            [CompletionResult]::new('--mode', '--mode', [CompletionResultType]::ParameterName, 'Output mode for chunk analysis')
            [CompletionResult]::new('--file-types', '--file-types', [CompletionResultType]::ParameterName, 'File extensions to analyze (comma-separated)')
            [CompletionResult]::new('-v', '-v', [CompletionResultType]::ParameterName, 'Verbosity level')
            [CompletionResult]::new('--verbose', '--verbose', [CompletionResultType]::ParameterName, 'Verbosity level')
            [CompletionResult]::new('--max-file-size', '--max-file-size', [CompletionResultType]::ParameterName, 'Maximum file size to analyze (bytes) - set to 0 to disable')
            [CompletionResult]::new('--max-lines', '--max-lines', [CompletionResultType]::ParameterName, 'Skip files with more than this many lines - set to 0 to disable')
            [CompletionResult]::new('--exclude', '--exclude', [CompletionResultType]::ParameterName, 'Exclude patterns (comma-separated, supports wildcards)')
            [CompletionResult]::new('--max-files', '--max-files', [CompletionResultType]::ParameterName, 'Maximum number of files to process (safety limit)')
            [CompletionResult]::new('--similarity', '--similarity', [CompletionResultType]::ParameterName, 'Similarity mode: none (disable), token (SimHash+Jaccard), embedding (fastembed/external)')
            [CompletionResult]::new('--sim-threshold', '--sim-threshold', [CompletionResultType]::ParameterName, 'Similarity threshold (Jaccard for token, cosine for embedding)')
            [CompletionResult]::new('--sim-top-k', '--sim-top-k', [CompletionResultType]::ParameterName, 'Top-k pairs to keep in the similarity report (0 = keep all)')
            [CompletionResult]::new('--sim-band-bits', '--sim-band-bits', [CompletionResultType]::ParameterName, 'Band size in bits for SimHash candidate bucketing')
            [CompletionResult]::new('--sim-min-tokens', '--sim-min-tokens', [CompletionResultType]::ParameterName, 'Minimum tokens required in a chunk to be considered for similarity')
            [CompletionResult]::new('--sim-output', '--sim-output', [CompletionResultType]::ParameterName, 'Write similarity report to this file (JSON)')
            [CompletionResult]::new('--embedder-cmd', '--embedder-cmd', [CompletionResultType]::ParameterName, 'Embedding command: "fastembed:<model>" or external process command')
            [CompletionResult]::new('--ann-k', '--ann-k', [CompletionResultType]::ParameterName, 'ANN neighbors (embedding mode)')
            [CompletionResult]::new('--ann-ef', '--ann-ef', [CompletionResultType]::ParameterName, 'ANN ef construction (embedding mode)')       
            [CompletionResult]::new('--ann-ef-search', '--ann-ef-search', [CompletionResultType]::ParameterName, 'ann-ef-search')
            [CompletionResult]::new('--ann-m', '--ann-m', [CompletionResultType]::ParameterName, 'ANN m (embedding mode)')
            [CompletionResult]::new('--verify-min-jaccard', '--verify-min-jaccard', [CompletionResultType]::ParameterName, 'Minimum token Jaccard to accept an embedding match (two-stage verify)')
            [CompletionResult]::new('--sim-print-limit', '--sim-print-limit', [CompletionResultType]::ParameterName, 'Limit number of similarity pairs to print (0 = all)')
            [CompletionResult]::new('--graph-output', '--graph-output', [CompletionResultType]::ParameterName, 'Build import dependency graph and write to this file (empty to skip)')
            [CompletionResult]::new('--api-backend-output', '--api-backend-output', [CompletionResultType]::ParameterName, 'Extract backend routes to this file (empty to skip)')
            [CompletionResult]::new('--api-frontend-output', '--api-frontend-output', [CompletionResultType]::ParameterName, 'Extract frontend calls to this file (empty to skip)')
            [CompletionResult]::new('--api-map-output', '--api-map-output', [CompletionResultType]::ParameterName, 'Map FE<->BE calls to this file (empty to skip)')
            [CompletionResult]::new('--suspects-output', '--suspects-output', [CompletionResultType]::ParameterName, 'Suspects (clusters + misplaced) output file')
            [CompletionResult]::new('--generate-completion', '--generate-completion', [CompletionResultType]::ParameterName, 'Generate shell completion scripts')
            [CompletionResult]::new('-r', '-r', [CompletionResultType]::ParameterName, 'Recursive analysis')
            [CompletionResult]::new('--recursive', '--recursive', [CompletionResultType]::ParameterName, 'Recursive analysis')
            [CompletionResult]::new('--force', '--force', [CompletionResultType]::ParameterName, 'Force processing of large files (ignores safety limits)')
            [CompletionResult]::new('--sim-cross-file-only', '--sim-cross-file-only', [CompletionResultType]::ParameterName, 'Only report cross-file similarities')
            [CompletionResult]::new('--sim-include-snippets', '--sim-include-snippets', [CompletionResultType]::ParameterName, 'Include code snippets in similarity report (and in stdout if --sim-print is set)')
            [CompletionResult]::new('--sim-print', '--sim-print', [CompletionResultType]::ParameterName, 'Print similarity pairs to stdout')     
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('-V', '-V ', [CompletionResultType]::ParameterName, 'Print version')
            [CompletionResult]::new('--version', '--version', [CompletionResultType]::ParameterName, 'Print version')
            break
        }
    })

    $completions.Where{ $_.CompletionText -like "$wordToComplete*" } |
        Sort-Object -Property ListItemText
}