import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';


let panel: vscode.WebviewPanel | undefined;

export async function activate(context: vscode.ExtensionContext) {

  
  // ✅ 1. Get Python extension and interpreter
  const pythonExt = vscode.extensions.getExtension("ms-python.python");
  let pythonPath: string | undefined = undefined;

  if (pythonExt) {
    if (!pythonExt.isActive) {
      await pythonExt.activate();
    }

    const pythonApi = pythonExt.exports;

    const execDetails = pythonApi.settings.getExecutionDetails();
    pythonPath = execDetails?.execCommand?.[0];

    console.log("🔍 Using Python interpreter:", pythonPath);
  } else {
    vscode.window.showWarningMessage("Python extension not found. Using system python.");
  }



  const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
  statusBarItem.text = '$(graph) Analyzer';
  statusBarItem.command = 'mytool.toggleGui';
  statusBarItem.tooltip = 'Toggle Code Analyzer GUI';
  statusBarItem.show();

  context.subscriptions.push(statusBarItem);

  const toggleCommand = vscode.commands.registerCommand('mytool.toggleGui', () => {
    if (panel) {
      panel.dispose();
      panel = undefined;
    } else {
      panel = vscode.window.createWebviewPanel(
        'codeAnalyzer',
        'Code Analyzer GUI',
        vscode.ViewColumn.One,
        { enableScripts: true }
      );

      const htmlPath = path.join(context.extensionPath, 'media', 'gui.html');
      const html = fs.readFileSync(htmlPath, 'utf8');
      panel.webview.html = html;

      panel.onDidDispose(() => {
        panel = undefined;
      });

      // Send active file info when webview loads
      const editor = vscode.window.activeTextEditor;
      if (editor && editor.document) {
        panel.webview.postMessage({
          command: 'setActiveFile',
          filePath: editor.document.uri.fsPath,
          fileName: path.basename(editor.document.uri.fsPath)
        });
      }

      panel.webview.onDidReceiveMessage(async message => {
        if (message.command === 'analyze') {
          // Get file path
          let filePath: string;
          let fileName: string;

          if (message.useActiveFile) {
            // Use active editor
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document) {
              filePath = editor.document.uri.fsPath;
              fileName = path.basename(filePath);
            } else {
              vscode.window.showErrorMessage('No active file found.');
              return;
            }
          } else {
            // Use provided file path
            filePath = message.fileName;
            fileName = path.basename(filePath);
          }

          // Notify webview that analysis started
          if (panel) {
            panel.webview.postMessage({
              command: 'analysisStarted',
              fileName: fileName,
              method: message.method
            });
          }

          // Execute Python analysis
          try {
            const results = await runQSpireAnalysis(filePath, message.method, pythonPath);
            
            // Send results back to webview
            if (panel) {
              panel.webview.postMessage({
                command: 'analysisComplete',
                results: results,
                fileName: fileName,
                method: message.method
              });
            }
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            vscode.window.showErrorMessage(`Analysis failed: ${errorMessage}`);
            
            if (panel) {
              panel.webview.postMessage({
                command: 'analysisError',
                error: errorMessage
              });
            }
          }
        }
      });
    }
  });

  context.subscriptions.push(toggleCommand);
}

/**
 * Run QSpire analysis using Python command

async function runQSpireAnalysis(filePath: string, method: string): Promise<any> {
  return new Promise((resolve, reject) => {
    // Determine which flag to use
    const flag = method === 'dynamic' ? '-dynamic' : '-static';
    
    // Get the workspace folder to determine the correct working directory
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    const cwd = workspaceFolder ? workspaceFolder.uri.fsPath : path.dirname(filePath);
    
    // Spawn qspire command
    // Note: Assumes qspire is installed and available in PATH (in venv)
    const pythonProcess = spawn('qspire', [flag, filePath], {
      cwd: cwd,
      shell: true  // Use shell to ensure venv activation is respected
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`QSpire exited with code ${code}: ${stderr}`));
        return;
      }

      try {
        // Parse the output to extract smells
        const smells = parseQSpireOutput(stdout, filePath);
        resolve({
          filePath: filePath,
          smells: {
            smells: smells
          }
        });
      } catch (error) {
        reject(error);
      }
    });

    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start QSpire: ${error.message}`));
    });
  });
}*/

async function runQSpireAnalysis(
  filePath: string,
  method: string,
  pythonPath?: string
): Promise<any> {

  return new Promise((resolve, reject) => {

    const flag = method === 'dynamic' ? '-dynamic' : '-static';

    // Workspace working directory
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    const cwd = workspaceFolder ? workspaceFolder.uri.fsPath : path.dirname(filePath);

    // Use selected venv Python, or fallback
    const pythonCmd = pythonPath || "python";

    console.log("🚀 Running QSpire with:", pythonCmd);

    // Run `python -m qspire -dynamic file.py`
    const pythonProcess = spawn(
      pythonCmd,
      ['-m', 'qspire', flag, filePath],
      {
        cwd,
        shell: false // NO NEED for shell when calling python directly
      }
    );

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`QSpire exited with code ${code}: ${stderr}`));
        return;
      }

      try {
        const smells = parseQSpireOutput(stdout, filePath);
        resolve({
          filePath: filePath,
          smells: { smells }
        });
      } catch (error) {
        reject(error);
      }
    });

    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start QSpire: ${error.message}`));
    });
  });
}


/**
 * Parse QSpire console output and extract smells
 */
function parseQSpireOutput(output: string, filePath: string): any {
  const smells: any[] = [];
  
  // Look for the results section
  const resultsMatch = output.match(/✅ Results:([\s\S]*?)={50}/);
  
  if (!resultsMatch) {
    // No results section found, return empty array
    return { smells: [] };
  }

  const resultsSection = resultsMatch[1];
  
  // Parse each smell (they appear as dictionaries)
  // Pattern to match dictionary-like structures
  const smellPattern = /\{[^}]+\}/g;
  const matches = resultsSection.match(smellPattern);
  
  if (matches) {
    for (const match of matches) {
      try {
        // Convert Python dict format to JSON
        // Replace single quotes with double quotes and handle Python-specific formats
        const jsonStr = match
          .replace(/'/g, '"')
          .replace(/None/g, 'null')
          .replace(/True/g, 'true')
          .replace(/False/g, 'false');
        
        const smell = JSON.parse(jsonStr);
        smells.push(smell);
      } catch (e) {
        console.error('Failed to parse smell:', match, e);
      }
    }
  }
  
  return { smells: smells };
}

export function deactivate() {}