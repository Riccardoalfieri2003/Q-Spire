import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';


/*
export function activate(context: vscode.ExtensionContext) {
  let disposable = vscode.commands.registerCommand('mytool.analyzeFile', async () => {
    // Step 1: Ask the user to pick a file from the workspace
    const uri = await vscode.window.showOpenDialog({
      canSelectMany: false,
      openLabel: 'Analyze This File',
      filters: {
        'Code files': ['js', 'py', 'ts', 'cpp', 'java', 'txt'] // adjust to your needs
      }
    });

    if (!uri || uri.length === 0) {
      vscode.window.showWarningMessage('No file selected.');
      return;
    }

    // Step 2: Read file content
    const filePath = uri[0].fsPath;
    const content = fs.readFileSync(filePath, 'utf8');

    // Step 3: Do something with it — for now, just show first 100 chars
    vscode.window.showInformationMessage(`First 100 characters:\n${content.slice(0, 100)}`);
  });

  context.subscriptions.push(disposable);
}
*/



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


/*
export function activate(context: vscode.ExtensionContext) {
  let disposable = vscode.commands.registerCommand('mytool.showGui', () => {
    const panel = vscode.window.createWebviewPanel(
      'codeAnalyzer',
      'Code Analyzer GUI',
      vscode.ViewColumn.One,
      {
        enableScripts: true
      }
    );

    const htmlPath = path.join(context.extensionPath, 'media', 'gui.html');
    const html = fs.readFileSync(htmlPath, 'utf8');
    panel.webview.html = html;
  });

  context.subscriptions.push(disposable);
}
*/