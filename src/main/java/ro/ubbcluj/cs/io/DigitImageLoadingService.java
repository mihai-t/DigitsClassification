package ro.ubbcluj.cs.io;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class DigitImageLoadingService {


    /**
     * the following constants are defined as per the values described at http://yann.lecun.com/exdb/mnist/
     **/
    private static final int OFFSET_SIZE = 4; //bytes

    private static final int LABEL_MAGIC = 2049;
    private static final int IMAGE_MAGIC = 2051;

    private static final int NUMBER_ITEMS_OFFSET = 4;
    private static final int ITEMS_SIZE = 4;

    private static final int NUMBER_OF_ROWS_OFFSET = 8;
    private static final int ROWS_SIZE = 4;
    private static final int ROWS = 28;

    private static final int NUMBER_OF_COLUMNS_OFFSET = 12;
    private static final int COLUMNS_SIZE = 4;
    private static final int COLUMNS = 28;

    private static final int IMAGE_OFFSET = 16;
    private static final int IMAGE_SIZE = ROWS * COLUMNS;
    private static final int CLASSES = 10;
    private final double notClassMark;
    private final double classMark;


    private String labelFileName;
    private String imageFileName;

    public DigitImageLoadingService(String labelFileName, String imageFileName, double classMark) {
        this.labelFileName = labelFileName;
        this.imageFileName = imageFileName;
        this.classMark = classMark;
        this.notClassMark = 0;
    }


    public List<TrainingExample> loadDigitImages() throws IOException {
        final ByteArrayOutputStream labelBuffer = new ByteArrayOutputStream();
        final ByteArrayOutputStream imageBuffer = new ByteArrayOutputStream();

        final InputStream labelInputStream = this.getClass().getResourceAsStream(labelFileName);
        final InputStream imageInputStream = this.getClass().getResourceAsStream(imageFileName);

        int read;
        final byte[] buffer = new byte[16000];

        while ((read = labelInputStream.read(buffer, 0, buffer.length)) != -1) {
            labelBuffer.write(buffer, 0, read);
        }

        while ((read = imageInputStream.read(buffer, 0, buffer.length)) != -1) {
            imageBuffer.write(buffer, 0, read);
        }

        final byte[] labelBytes = labelBuffer.toByteArray();
        final byte[] imageBytes = imageBuffer.toByteArray();


        final byte[] labelMagic = Arrays.copyOfRange(labelBytes, 0, OFFSET_SIZE);
        final byte[] imageMagic = Arrays.copyOfRange(imageBytes, 0, OFFSET_SIZE);

        if (ByteBuffer.wrap(labelMagic).getInt() != LABEL_MAGIC) {
            throw new IOException("Bad magic number in label file!");
        }

        if (ByteBuffer.wrap(imageMagic).getInt() != IMAGE_MAGIC) {
            throw new IOException("Bad magic number in image file!");
        }

        final int numberOfLabels = ByteBuffer.wrap(Arrays.copyOfRange(labelBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();
        final int numberOfImages = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_ITEMS_OFFSET, NUMBER_ITEMS_OFFSET + ITEMS_SIZE)).getInt();

        if (numberOfImages != numberOfLabels) {
            throw new IOException("The number of labels and images do not match!");
        }


        final int numRows = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_ROWS_OFFSET, NUMBER_OF_ROWS_OFFSET + ROWS_SIZE)).getInt();
        final int numCols = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, NUMBER_OF_COLUMNS_OFFSET, NUMBER_OF_COLUMNS_OFFSET + COLUMNS_SIZE)).getInt();

        if (numRows != ROWS && numCols != COLUMNS) {
            throw new IOException("Bad image. Rows and columns do not equal " + ROWS + "x" + COLUMNS);
        }

        final List<TrainingExample> trainingExamples = new ArrayList<>();

        for (int i = 0; i < numberOfLabels; i++) {
            final int label = labelBytes[OFFSET_SIZE + ITEMS_SIZE + i];

            final double[] features = new double[IMAGE_SIZE];
            for (int j = (i * IMAGE_SIZE) + IMAGE_OFFSET; j < (i * IMAGE_SIZE) + IMAGE_OFFSET + IMAGE_SIZE; ++j) {
                final int unsignedByte = ((int) imageBytes[j]) & 0xFF;
                final double e = (unsignedByte) / (255.0);//min-max normalization
                features[j - (i * IMAGE_SIZE) - IMAGE_OFFSET] = e;
            }

            double[] target = new double[CLASSES];
            for (int j = 0; j < CLASSES; ++j) {
                target[j] = notClassMark;
            }
            target[label] = classMark;

            final TrainingExample example = new TrainingExample(features, target);

            trainingExamples.add(example);
        }

        return trainingExamples;
    }


}
