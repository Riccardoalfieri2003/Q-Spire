/*
import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

let panel: vscode.WebviewPanel | undefined;

export function activate(context: vscode.ExtensionContext) {
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

      panel.webview.onDidReceiveMessage(async message => {
        if (message.command === 'analyze') {
          // 1. Try to get the active editor
          const editor = vscode.window.activeTextEditor;

          if (editor && editor.document) {
            const document = editor.document;
            const filePath = document.uri.fsPath;
            const content = document.getText();

            vscode.window.showInformationMessage(`Using active file: ${filePath}`);
            // Do your analysis here
            console.log("Analyzing active file:", content.slice(0, 100));
            // You can also send content back to the WebView if needed

          } else {
            // 2. Ask the user to pick a file
            const uri = await vscode.window.showOpenDialog({
              canSelectMany: false,
              openLabel: 'Choose file to analyze',
              filters: {
                'Code files': ['py'],
              }
            });

            if (!uri || uri.length === 0) {
              vscode.window.showWarningMessage('No file selected.');
              return;
            }

            const filePath = uri[0].fsPath;
            const content = fs.readFileSync(filePath, 'utf8');

            vscode.window.showInformationMessage(`Using selected file: ${filePath}`);
            console.log("Analyzing selected file:", content.slice(0, 100));
          }
        }
      });


    }
  });

  context.subscriptions.push(toggleCommand);
}
*/

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';


let panel: vscode.WebviewPanel | undefined;

export function activate(context: vscode.ExtensionContext) {
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
            const results = await runQSpireAnalysis(filePath, message.method);
            
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
 */
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