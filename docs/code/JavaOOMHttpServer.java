import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import javassist.ClassPool;
import javassist.CtClass;
import javassist.Loader;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;


public class JavaOOMHttpServer {
    static class Key{
        int x=1;
        short y=2;
    }

    static class HeapKey {
        Integer id;

        HeapKey(Integer id) {
            this.id = id;
        }

        @Override
        public int hashCode() {
            return id.hashCode();
        }
    }

    // Execute it and go to http://localhost:8000/*
    public static void main(String[] args) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);
        server.createContext("/gcoverhead", new GCOverheadHandler());
        server.createContext("/heapspace", new HeapspaceHandler());
        server.createContext("/metaspace", new MetaspaceHandler());
        server.createContext("/thread", new ThreadHandler());
        server.createContext("/buffer", new DirectBufferHandler());
        server.setExecutor(null); // creates a default executor
        server.start();
    }

    public static void send_response(HttpExchange exchange, String type) throws IOException {
        String response = String.format("Start fault injection: %s", type);
        exchange.sendResponseHeaders(200, response.length());
        OutputStream os = exchange.getResponseBody();
        os.write(response.getBytes());
        os.close();
    }

    public static void GCOverhead() throws Throwable {
        System.out.println("Start.....");
        ArrayList<Key> list = new ArrayList<Key>();
        try {
            for (int l = 0; l < 20; l++) {
                try {
                    for (int i = 0; ; i++) {
                        list.add(new Key());
                        if (i % 100000 == 0) {
                            System.out.println(i);
                            TimeUnit.SECONDS.sleep(1);
                        }
                    }
                } catch (Throwable t) {
                    System.out.println("Catch exception1");
                    t.printStackTrace();
                }
                TimeUnit.SECONDS.sleep(3);
            }
        } catch (Throwable t) {
            System.out.println("Catch exception1");
            t.printStackTrace();
        }
        System.out.println("GCOverhead End........");
    }

    public static void Heapspace() throws Throwable {
        try {
            Map<HeapKey, String> m = new HashMap<HeapKey, String>();
            for (int l = 0; l < 20; l++)
            {
                try {
                    for (int i =0; ; i++) {
                        if (!m.containsKey(new HeapKey(i))) {
                            m.put(new HeapKey(i), "Number:" + i);
                        }
                        if (i % 50000 == 0) {
                            System.out.println(i);
                            TimeUnit.SECONDS.sleep(1);
                        }
                    }
                } catch (Throwable t) {
                    System.out.println("Catch exception1");
                    t.printStackTrace();
                }
                TimeUnit.SECONDS.sleep(10);
            }
        } catch (Throwable t)
        {
            System.out.println("Catch exception2");
            t.printStackTrace();
        }
        System.out.println("Heapspace End.....");
    }

    public static int count = 0;

    public static void Metaspace() throws Throwable {
        ClassPool cp = new ClassPool(true);
        Loader cl = new Loader(cp);
        int i = 0;
        try {
//            for (int l = 0; l < 5; l++) {
//                try {
                    while (true) {
                        i++;
                        CtClass ct = cp.makeClass("Generated" + (count + i));
                        cl.loadClass("Generated" + (count + i));
                        if ((count + i) % 10000 == 0) {
                            System.out.println((count + i));
                            TimeUnit.SECONDS.sleep(1);
                        }
                    }
//                } catch (Throwable t) {
//                    t.printStackTrace();
//                }
//                TimeUnit.SECONDS.sleep(1);
//            }
        } catch (Throwable t)
        {
            t.printStackTrace();
        }

        cp = null;
        cl = null;
        count += i;
        System.out.println("Metaspace End........");
    }

    public static void Thread() throws Throwable {
        try {
//            for (int l = 0; l < 10; l++) {
                try {
                    while (true) {
                        new Thread(new Runnable() {
                            public void run() {
                                try {
                                    TimeUnit.SECONDS.sleep(150);
                                } catch (Throwable t) {
                                    t.printStackTrace();
                                }
                            }
                        }).start();
                        Thread.sleep(2);
                    }
                } catch (Throwable t) {
                    t.printStackTrace();
                }

                TimeUnit.SECONDS.sleep(100);
//            }
        } catch (Throwable t)
        {
            t.printStackTrace();
        }
        System.out.println("Thread End........");
    }

    public static void Buffer() throws InterruptedException {
        List<ByteBuffer> buffers = new ArrayList<ByteBuffer>();
        try {
            int i = 0;
            while (true) {
                ByteBuffer tmp = ByteBuffer.allocateDirect(i);
                buffers.add(tmp);
                i++;

                TimeUnit.MILLISECONDS.sleep(1);
            }
        } catch (Throwable t) {
            t.printStackTrace();
        }
        System.out.println("Direct Buffer End........");
    }

    static class GCOverheadHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            send_response(exchange, "GCOverhead");
            try {
                GCOverhead();
            } catch (Throwable t)
            {
                t.printStackTrace();
            }
        }
    }

    static class HeapspaceHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            send_response(exchange, "Heapspace");
            try {
                Heapspace();
            } catch (Throwable t)
            {
                t.printStackTrace();
            }

        }
    }

    static class MetaspaceHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            send_response(exchange, "Metaspace");
            try {
                Metaspace();
            } catch (Throwable t)
            {
                t.printStackTrace();
            }
        }
    }

    static class ThreadHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            send_response(exchange, "Thread");
            try {
                Thread();
            } catch (Throwable t)
            {
                t.printStackTrace();
            }
        }
    }

    static class DirectBufferHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            send_response(exchange, "Buffer");
            try {
                Buffer();
            } catch (Throwable t)
            {
                t.printStackTrace();
            }
        }
    }
}
