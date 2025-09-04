import { NextRequest, NextResponse } from 'next/server';
import { writeFile, unlink } from 'fs/promises';
import { join } from 'path';
import { spawn } from 'child_process';

const UPLOAD_DIR = join(process.cwd(), 'uploads');
const SCRIPTS_DIR = join(process.cwd(), 'app', 'scripts');

// Ensure upload directory exists
async function ensureUploadDir() {
  try {
    await import('fs').then(fs => fs.promises.mkdir(UPLOAD_DIR, { recursive: true }));
  } catch (error) {
    // Directory might already exist
  }
}

export async function POST(request: NextRequest) {
  try {
    await ensureUploadDir();

    const formData = await request.formData();
    const file = formData.get('video') as File;
    const effect = formData.get('effect') as string;
    const objectType = formData.get('objectType') as string;

    if (!file || !effect) {
      return NextResponse.json(
        { error: 'Missing video file or effect' },
        { status: 400 }
      );
    }

    // Validate object type for specific effects
    if (effect === 'object_specific_blobs' && !objectType?.trim()) {
      return NextResponse.json(
        { error: 'Object type is required for the Specific Blobs effect' },
        { status: 400 }
      );
    }

    // Generate unique filenames
    const timestamp = Date.now();
    const inputFilename = `input_${timestamp}_${file.name}`;
    const outputFilename = `output_${timestamp}.mp4`;
    const inputPath = join(UPLOAD_DIR, inputFilename);
    const outputPath = join(UPLOAD_DIR, outputFilename);

    // Save uploaded file
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    await writeFile(inputPath, buffer);

    // Map effect to script
    const scriptMap: { [key: string]: string } = {
      'blob_detect': 'blob_detect.py',
      'cryptomatte_video': 'cryptomatte_video.py',
      'dotpad': 'dotpad.py',
      'object_connectors_video': 'object_connectors_video.py',
      'object_specific_blobs': 'object_specific_blobs.py',
      'ruttra_video': 'ruttra_video.py',
      'splinterframes': 'splinterframes.py',
    };

    const scriptName = scriptMap[effect];
    if (!scriptName) {
      await unlink(inputPath);
      return NextResponse.json(
        { error: 'Invalid effect selected' },
        { status: 400 }
      );
    }

    const scriptPath = join(SCRIPTS_DIR, scriptName);

    // Prepare arguments for Python script
    const scriptArgs = [scriptPath, inputPath, outputPath];
    
    // Add object type parameter for specific effects
    if (effect === 'object_specific_blobs' && objectType?.trim()) {
      scriptArgs.push(objectType.trim());
    }

    // Execute Python script
    await new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', scriptArgs, {
        cwd: process.cwd(),
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
        if (code === 0) {
          resolve(stdout);
        } else {
          console.error('Python script error:', stderr);
          reject(new Error(`Script failed with code ${code}: ${stderr}`));
        }
      });

      pythonProcess.on('error', (error) => {
        console.error('Failed to start python process:', error);
        reject(error);
      });
    });

    // Read the processed video file
    const processedVideo = await import('fs').then(fs => fs.promises.readFile(outputPath));

    // Clean up temporary files
    try {
      await unlink(inputPath);
      await unlink(outputPath);
    } catch (error) {
      console.error('Error cleaning up files:', error);
    }

    // Return the processed video
    return new NextResponse(processedVideo, {
      status: 200,
      headers: {
        'Content-Type': 'video/mp4',
        'Content-Disposition': `attachment; filename="processed_${file.name}"`,
      },
    });

  } catch (error) {
    console.error('Error processing video:', error);
    return NextResponse.json(
      { error: 'Failed to process video' },
      { status: 500 }
    );
  }
}
