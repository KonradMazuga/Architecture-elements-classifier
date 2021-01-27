package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class SecondActivity extends AppCompatActivity {

    String[] labels = {"altar", "apse", "bell_tower", "column","dome(inner)", "dome(outer)", "flying_buttress", "gargoyle","stained_glass","vault"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);
        Intent photo = getIntent();
        Bundle data = photo.getBundleExtra("data_raw");

        Bitmap image = (Bitmap)data.get("data");
        int w=image.getWidth();
        int h=image.getHeight();
        int crop = Math.abs((w-h)/2);
        Bitmap image_new=Bitmap.createBitmap(image,crop,crop,w-2*crop,w-2*crop,null,true);
        Bitmap resizedBitmap=Bitmap.createScaledBitmap(image_new,128,128,true);

        ImageView imageView = (ImageView)findViewById(R.id.imageView);
        imageView.setImageBitmap(resizedBitmap);

        TextView text = (TextView)findViewById(R.id.textView);
        Interpreter interpreter = null;

        try {
            ByteBuffer mappedByteBuffer = loadModelFile();
            Interpreter.Options options = new Interpreter.Options();
            interpreter = new Interpreter(mappedByteBuffer,options);
        }
        catch (IOException ex) {
            ex.printStackTrace();
            return;
        }

        ByteBuffer in = ByteBuffer.allocateDirect(4 * 128 * 128 * 3 );
        in.order(ByteOrder.nativeOrder());
        for (int y = 0; y < 128; y++) {
            for (int x = 0; x < 128; x++) {
                int pixel = resizedBitmap.getPixel(x, y);
                int r = Color.red(pixel);
                int g = Color.green(pixel);
                int b = Color.blue(pixel);
                in.putFloat(r);
                in.putFloat(g);
                in.putFloat(b);
            }
        }

        ByteBuffer out = ByteBuffer.allocateDirect(10 * 4);
        out.order(ByteOrder.nativeOrder());
        interpreter.run(in, out);
        out.rewind();
        FloatBuffer score = out.asFloatBuffer();

        float max_val = 0;
        String max_label = null;
        for (int i = 0; i < 10; i++) {
            String label = labels[i];
            float p = score.get(i);
            if (p > max_val) {
                max_val = p;
                max_label = label;
            }

        }
        text.setText("It is a/an: "+ max_label);



    }

    private MappedByteBuffer loadModelFile() throws IOException{
        AssetFileDescriptor fileDescriptor=this.getAssets().openFd("architecture_model.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }


}