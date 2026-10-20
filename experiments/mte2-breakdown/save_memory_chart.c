/* Preserve the tool's parsed DBI records before its normal cleanup.
 * Host filesystem observation only; no changes to launches or device code. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

static void preserve(int dirfd, const char *path) {
    int saved_errno = errno;
    const char *out = getenv("MEMORY_CHART_SAVE_DIR");
    const char *base = path ? strrchr(path, '/') : NULL;
    base = base ? base + 1 : path;
    if (!out || !base || strcmp(base, "MemoryChart.bin")) return;
    int in = openat(dirfd, path, O_RDONLY);
    if (in < 0) return;
    char dst[4096];
    snprintf(dst, sizeof(dst), "%s/MemoryChart-%d.bin", out, getpid());
    int fd = open(dst, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) {
        char buf[65536]; ssize_t n;
        while ((n = read(in, buf, sizeof(buf))) > 0) {
            ssize_t off = 0;
            while (off < n) { ssize_t w = write(fd, buf + off, n - off); if (w <= 0) break; off += w; }
            if (off != n) break;
        }
        close(fd);
    }
    close(in);
    errno = saved_errno;
}
int unlink(const char *path) {
    static int (*next)(const char *);
    if (!next) next = dlsym(RTLD_NEXT, "unlink");
    preserve(AT_FDCWD, path); return next(path);
}
int unlinkat(int dirfd, const char *path, int flags) {
    static int (*next)(int, const char *, int);
    if (!next) next = dlsym(RTLD_NEXT, "unlinkat");
    preserve(dirfd, path); return next(dirfd, path, flags);
}
int remove(const char *path) {
    static int (*next)(const char *);
    if (!next) next = dlsym(RTLD_NEXT, "remove");
    preserve(AT_FDCWD, path); return next(path);
}
